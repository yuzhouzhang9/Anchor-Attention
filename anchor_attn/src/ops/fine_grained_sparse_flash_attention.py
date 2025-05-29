import torch
import triton
import triton.language as tl
import time
from my_utils import Logger

# ------------------------------
# Kernel Definition
# ------------------------------
@triton.jit
def _triton_fine_grained_sparse_attn_fwd_kernel(
    Q, K, V, sm_scale,
    true_coords, true_counts,
    Out, L_buffer, M_buffer, Acc_buffer,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_mb, stride_mh,
    stride_abz, stride_abh, stride_abm, stride_abd,
    stride_tcz, stride_tch, stride_tcm, stride_tck,
    stride_tctz, stride_tcth, stride_tctm, stride_tctk,
    Z, H, N_CTX,
    STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_INDICES: tl.constexpr,
    dtype: tl.constexpr,
):
    # Program IDs
    start_m = tl.program_id(0)
    big_block_startm = (start_m * BLOCK_M) // STEP
    off_hz = tl.program_id(1)
    query_start = start_m * BLOCK_M
    offs_m = query_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    lm_offset = (off_hz // H) * stride_lb + (off_hz % H) * stride_lh
    ab_offset = (off_hz // H) * stride_abz + (off_hz % H) * stride_abh
    # coords_cnt/coords [B, H, big_blocks, 1 / N]
    # grid [qblocks, B * H]
    tc_offset = (off_hz // H) * stride_tcz + (off_hz % H) * stride_tch + big_block_startm * stride_tcm
    tct_offset = (off_hz // H) * stride_tctz + (off_hz % H) * stride_tcth + big_block_startm * stride_tctm 
    # q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    # o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    # tc_ptrs = true_coords + tc_offset
    tct_ptrs = true_counts + tct_offset
    qk_scale = sm_scale * 1.44269504
    sparse_count = tl.load(tct_ptrs) 
    offs_m = query_start + tl.arange(0, BLOCK_M) 
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    l_ptrs = L_buffer + lm_offset + offs_m
    m_ptrs = M_buffer + lm_offset + offs_m
    a_ptrs = Acc_buffer + ab_offset + offs_m[:, None] * stride_abm + offs_d[None, :] * stride_abd
    m_i = tl.load(m_ptrs)
    l_i = tl.load(l_ptrs)
    acc = tl.load(a_ptrs)
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)
    tc_ptrs = true_coords + tc_offset
    for idx_start in range(0, sparse_count, NUM_INDICES):
        offs_idx = idx_start + tl.arange(0, NUM_INDICES)
        idx_mask = offs_idx < sparse_count
        real_block_idx = tl.load(tc_ptrs + offs_idx, mask=idx_mask, other=0)
        offs_n = real_block_idx
        combined_mask = idx_mask
        k_ptrs_group = k_ptrs + offs_n[None, :] * stride_kn
        v_ptrs_group = v_ptrs + offs_n[:, None] * stride_vn
        k = tl.load(k_ptrs_group, mask=combined_mask[None, :], other=0.0)
        v = tl.load(v_ptrs_group, mask=combined_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, NUM_INDICES], dtype=dtype)
        qk += tl.dot(q, k)
        qk = tl.where(combined_mask[None, :], qk, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype))

# ------------------------------
# Helper Function: Generate Sparse Coordinates
# ------------------------------
def generate_sparse_coords_and_counts(B, H, N, block_size, step, ratio, device, max_blocks_per_row=64):
    """
    Generate true_coords and true_counts tensors by sampling random key block coordinates
    based on a target sparsity ratio.
    
    Args:
        B (int): Batch size.
        H (int): Number of attention heads.
        N (int): Sequence length.
        block_size (int): Size of each block.
        step (int): Number of blocks per query group.
        ratio (float): Target sparsity ratio (0 to 1).
        device (torch.device): Device for tensors.
        max_blocks_per_row (int): Maximum number of key blocks per query group.
    
    Returns:
        true_coords (torch.Tensor): Shape (B, H, NUM_ROWS_STEP2, max_blocks_per_row).
        true_counts (torch.Tensor): Shape (B, H, NUM_ROWS_STEP2, 1).
    """
    NUM_ROWS = N // block_size
    NUM_ROWS_STEP2 = (NUM_ROWS + step - 1) // step
    MAX_BLOCKS_PER_ROW = max_blocks_per_row
    true_coords = torch.zeros((B, H, NUM_ROWS_STEP2, MAX_BLOCKS_PER_ROW), dtype=torch.int32, device=device)
    true_counts = torch.zeros((B, H, NUM_ROWS_STEP2, 1), dtype=torch.int32, device=device)
    Logger.log(f"true_coords:{true_coords.shape}")
    if ratio == 1.0:
        # For ratio=1.0, include all possible key block indices (causal attention)
        for b in range(B):
            for h in range(H):
                for i in range(NUM_ROWS_STEP2):
                    # Determine the range of valid key blocks for query group i
                    # Query group i corresponds to query blocks [i*step, (i+1)*step)
                    # Key blocks k <= query block index (causal attention)
                    max_key_block = (i + 1) * step * block_size# Up to the current query block
                    num_valid_blocks = min(max_key_block, MAX_BLOCKS_PER_ROW)  # Respect max_blocks_per_row
                    if num_valid_blocks > 0:
                        # Fill true_coords with indices [0, 1, ..., num_valid_blocks-1]
                        true_coords[b, h, i, :num_valid_blocks] = torch.arange(
                            num_valid_blocks, dtype=torch.int32, device=device
                        )
                        true_counts[b, h, i, 0] = num_valid_blocks
        Logger.log(f"Ratio 1.0: Filled all possible key block indices")
        return true_coords, true_counts
    total_possible_blocks = B * H * (NUM_ROWS_STEP2 + 1) * (N // 2)
    total_selected_blocks = int(ratio * total_possible_blocks)

    if total_selected_blocks == 0:
        return true_coords, true_counts


    coords = torch.randperm(total_possible_blocks, device=device)[:total_selected_blocks]
    
    l = 0
    r = 0
    used = 0
    delta = N // NUM_ROWS_STEP2
    for b in range(B):
        for h in range(H):
            cur = 0
            for i in range(NUM_ROWS_STEP2):
                cur += delta
                r = l + cur
                mask = (coords >= l) & (coords <= r)
                selected_coords = coords[mask]
                num_selected = selected_coords.size(0)
                if num_selected > 0:
                    k_indices = (selected_coords - l)
                    true_coords[b, h, i, :num_selected] = k_indices[:num_selected]
                    true_counts[b, h, i, 0] = num_selected
                used += num_selected
                l = r
    Logger.log(f"Ratio {ratio:.2f}: Used {used}/{total_selected_blocks} blocks")
    true_counts = true_counts.clamp(0, MAX_BLOCKS_PER_ROW)
    return true_coords, true_counts

# ------------------------------
# Helper Function: Initialize Inputs
# ------------------------------
def initialize_inputs(B, H, N, D, block_size, device, dtype=torch.bfloat16):
    """
    Initialize input tensors and buffers for the kernel.
    
    Args:
        B (int): Batch size.
        H (int): Number of attention heads.
        N (int): Sequence length.
        D (int): Model dimension.
        block_size (int): Size of each block.
        device (str): Device for tensors.
        dtype (torch.dtype): Data type for tensors.
    
    Returns:
        dict: Dictionary containing Q, K, V, seqlens, sm_scale, output, L_buffer, M_buffer, Acc_buffer.
    """
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous() * 2
    K = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous() * 2
    V = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous() / 2
    seqlens = torch.full((B,), N, dtype=torch.int32, device=device)
    sm_scale = 1.0 / (D ** 0.5)
    output = torch.zeros_like(Q, device=device)
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=device)
    M_buffer = torch.full((B, H, N), -float("inf"), dtype=torch.float32, device=device)
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32, device=device)
    
    return {
        "Q": Q,
        "K": K,
        "V": V,
        "seqlens": seqlens,
        "sm_scale": sm_scale,
        "output": output,
        "L_buffer": L_buffer,
        "M_buffer": M_buffer,
        "Acc_buffer": Acc_buffer
    }

# ------------------------------
# Test Function
# ------------------------------
def test_fine_grained_kernel(B=1, H=32, N=1024*128, D=128, step=16, num_runs=2, device="cuda:0"):
    """
    Test the _triton_fine_grained_sparse_attn_fwd_kernel for different sparsity ratios.
    
    Args:
        B (int): Batch size.
        H (int): Number of attention heads.
        N (int): Sequence length.
        D (int): Model dimension.
        step (int): Number of blocks per query group.
        num_runs (int): Number of runs per sparsity ratio.
        device (str): Device for tensors.
    """
    Logger.set_log_file_path(f"log/test_fine_grained/{time.time()}.log")
    block_size = 128
    dtype = torch.bfloat16
    NUM_ROWS = N // block_size
    # NUM_ROWS_STEP2 = (NUM_ROWS + step - 1) // step

    # Initialize inputs
    inputs = initialize_inputs(B, H, N, D, block_size, device, dtype)
    Q, K, V = inputs["Q"], inputs["K"], inputs["V"]
    seqlens, sm_scale = inputs["seqlens"], inputs["sm_scale"]
    output = inputs["output"]
    L_buffer, M_buffer, Acc_buffer = inputs["L_buffer"], inputs["M_buffer"], inputs["Acc_buffer"]
    # Test across sparsity ratios
    for ratio in range(0, 105, 5):
        ratio /= 100
        Logger.log(f"\nTesting sparsity ratio: {ratio:.2f}")

        # Generate sparse coordinates
        true_coords, true_counts = generate_sparse_coords_and_counts(
            B, H, N, block_size, step, ratio, device, max_blocks_per_row=N
        )
        MAX_BLOCKS_PRE_ROW_STEP2 = true_coords.shape[-1]

        # Kernel grid
        grid = (NUM_ROWS, B * H)

        # Run the kernel multiple times
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            _triton_fine_grained_sparse_attn_fwd_kernel[grid](
                Q, K, V, sm_scale,
                true_coords, true_counts,
                output, L_buffer, M_buffer, Acc_buffer,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                L_buffer.stride(0), L_buffer.stride(1),
                M_buffer.stride(0), M_buffer.stride(1),
                Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),
                true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
                true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
                B, H, N,
                STEP=step * block_size,
                BLOCK_M=128,
                BLOCK_DMODEL=D,
                NUM_INDICES=128,
                dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
                num_warps=8,
                num_stages=3
            )

            torch.cuda.synchronize()
            elapsed_time = (time.time() - start_time) * 1000
            Logger.log(f"Run {i+1}, Ratio {ratio:.2f}: {elapsed_time:.2f} ms")

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    test_fine_grained_kernel(B=1, H=1, N=1024*128, D=128, step=16, num_runs=3)