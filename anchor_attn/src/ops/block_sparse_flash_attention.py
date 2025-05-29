import torch
import triton
import triton.language as tl
import time
from flash_attn import flash_attn_func
from my_utils import Logger

# ------------------------------
# Kernel Definition
# ------------------------------
@triton.jit
def _triton_block_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)



@triton.jit
def _triton_block_sparse_attn_with_count_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_index, block_index_context,Out,# Added Acc_buffer
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_bic_z, stride_bic_h, stride_bic_m,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW
    bic_ptr = block_index_context + (off_hz // H) * stride_bic_z + (off_hz % H) * stride_bic_h + start_m * stride_bic_m

    # Initialize m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Scale sm_scale
    qk_scale = sm_scale * 1.44269504

    # Load Q
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # Mask
    m_mask = offs_m[:, None] < seqlen

    # Load block_count from block_index_context
    block_count = tl.load(bic_ptr)
    block_count = tl.minimum(block_count, MAX_BLOCKS_PRE_ROW)

    # Iterate over sparse blocks
    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n

        # Load K and V
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)

        # Compute scaling constants
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # Scale and update accumulator
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)

        # Update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Normalize output
    acc /= l_i[:, None]
    
    # Write output O
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


# ------------------------------
# Block Sparse Attention Function
# ------------------------------
def _triton_block_sparse_attention(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PRE_ROW]
    sm_scale,
    block_size_M=64,
    block_size_N=64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_block_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_index.shape[-2], block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )
    return o

# ------------------------------
# Build Block Index Function
# ------------------------------
def _build_block_index(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)
    arange_M = torch.arange(query_pool.shape[-2], dtype=torch.int32, device=query.device) * block_size_M
    arange_N = torch.arange(key_pool.shape[-2], dtype=torch.int32, device=key.device) * block_size_N
    p_pool = torch.einsum('bhmk,bhnk->bhmn', query_pool, key_pool)
    p_pool = p_pool.where(arange_M[None, None, :, None] >= arange_N[None, None, None, :], -torch.inf)
    top_k = min(top_k, context_size // block_size_N)
    return torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values

# ------------------------------
# Block Sparse Attention Wrapper
# ------------------------------
def block_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = block_size_M - (query.shape[2] % block_size_M)
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    block_index = _build_block_index(query, key, top_k, block_size_M, block_size_N)
    out = _triton_block_sparse_attention(query, key, value, seqlens, block_index, sm_scale, block_size_M, block_size_N)
    return out[..., :context_size, :]

    
# ------------------------------
# Random Block Index Generation
# ------------------------------
def generate_random_block_index(B, H, N, block_size_M, block_size_N, ratio, device):
    """
    Generate a block_index tensor with random key block indices based on a target sparsity ratio.
    
    Args:
        B (int): Batch size.
        H (int): Number of attention heads.
        N (int): Sequence length.
        block_size_M (int): Size of query block.
        block_size_N (int): Size of key/value block.
        ratio (float): Target sparsity ratio (0 to 1).
        device (torch.device): Device for tensors.
        max_blocks_per_row (int): Maximum number of key blocks per query row.
    
    Returns:
        block_index (torch.Tensor): Shape (B, H, NUM_ROWS, max_blocks_per_row).
    """
    NUM_ROWS = (N + block_size_M - 1) // block_size_M  # Ceiling division
    NUM_BLOCKS_N = (N + block_size_N - 1) // block_size_N  # Total key blocks

    block_index = torch.zeros((B, H, NUM_ROWS, NUM_BLOCKS_N), dtype=torch.int32, device=device)
    block_count = torch.zeros((B, H, NUM_ROWS, 1), dtype=torch.int32, device=device)
    if ratio == 1.0:
        # Include all possible key block indices (causal attention)
        for b in range(B):
            for h in range(H):
                for m in range(NUM_ROWS):
                    # For query block m, valid key blocks are k <= m
                    max_key_block = m + 1
                    num_valid_blocks = min(max_key_block, NUM_BLOCKS_N)
                    if num_valid_blocks > 0:
                        block_index[b, h, m, :num_valid_blocks] = torch.arange(
                            num_valid_blocks, dtype=torch.int32, device=device
                        )
                        block_count[b, h, m, 0] = num_valid_blocks
        Logger.log(f"Ratio 1.0: Filled all possible key block indices")
        return block_index,block_count

    # For ratio < 1.0, use random sampling
    total_possible_blocks = B * H * NUM_ROWS * (NUM_BLOCKS_N + 1) // 2  # Approximate for causal mask
    total_selected_blocks = int(ratio * total_possible_blocks)

    if total_selected_blocks == 0:
        Logger.log(f"Ratio {ratio:.2f}: No blocks selected")
        return block_index

    coords = torch.randperm(total_possible_blocks, device=device)[:total_selected_blocks]

    l = 0
    r = 0
    used = 0
    for b in range(B):
        for h in range(H):
            for m in range(NUM_ROWS):
                # For query block m, valid key blocks are k <= m
                max_key_block = min(m + 1, NUM_BLOCKS_N)
                cur = max_key_block
                r = l + cur
                mask = (coords >= l) & (coords < r)
                selected_coords = coords[mask]
                num_selected = selected_coords.size(0)
                if num_selected > 0:
                    k_indices = (selected_coords - l)
                    block_index[b, h, m, :num_selected] = k_indices[:num_selected]
                    block_count[b, h, m, 0] = num_selected
                used += num_selected
                l = r
    Logger.log(f"Ratio {ratio:.2f}: Used {used}/{total_selected_blocks} blocks")
    return block_index,block_count

# ------------------------------
# Test Function for Random Block Indices
# ------------------------------
def test_block_sparse_attention_random(B=1, H=4, N=128*1024, D=128, block_size_M=128, block_size_N=128, num_runs=3, device="cuda:0"):
    """
    Test the _triton_block_sparse_attention with random block indices for different sparsity ratios.
    
    Args:
        B (int): Batch size.
        H (int): Number of attention heads.
        N (int): Sequence length.
        D (int): Model dimension.
        block_size_M (int): Size of query block.
        block_size_N (int): Size of key/value block.
        num_runs (int): Number of runs per sparsity ratio.
        device (str): Device for tensors.
    """
    # Logger.set_log_file_path(f"log/test_block_random/{time.time()}.log")
    dtype = torch.bfloat16

    # Initialize inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    k = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    v = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    seqlens = torch.tensor([N], dtype=torch.int32, device=device)
    sm_scale = D ** -0.5

    # Test across sparsity ratios
    for ratio in list(range(5, 100, 5)) + [100]:  # Include 1.0
        ratio /= 100
        Logger.log(f"\nTesting sparsity ratio: {ratio:.2f}")

        # Generate random block indices
        block_index,block_count = generate_random_block_index(
            B, H, N, block_size_M, block_size_N, ratio, device
        )
        Logger.log(f"block_index:{block_index.shape}")
        # Run the kernel multiple times
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            output = torch.zeros_like(q)
            NUM_ROWS = N//block_size_M
            grid = (NUM_ROWS, B * H)
            _triton_block_sparse_attn_with_count_fwd_kernel[grid](
                q, k, v, seqlens, sm_scale,
                block_index, block_count,output,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                block_count.stride(0), block_count.stride(1), block_count.stride(2),
                B, H, N,
                NUM_ROWS, 
                block_index.shape[-1],
                BLOCK_M=block_size_M,
                BLOCK_N=block_size_N,
                BLOCK_DMODEL=D,
                dtype=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
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
    import time
    import torch
    
    # Set up logging
    Logger.set_log_file_path(f"log/test_block/{time.time()}.log")
    
    # Test parameters
    B = 1
    H = 1
    N = 128 * 1024   # Sequence length
    D = 128             # Head dimension
    block_size_M = 128
    block_size_N = 128
    num_runs = 2
    device = "cuda:0"
    dtype = torch.bfloat16
    
    # Initialize random input tensors
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous()
    
    # Test: Block Sparse Attention (top_k, top_cdf, top_difference_aware)
    Logger.log("\nTesting block_sparse_attention, block_sparse_attention_top_cdf, and block_sparse_attention_top_difference_aware")
    top_k = 64 # Select half of the key blocks (32,768)
    cdf_threshold = 0.95            # CDF threshold
    theta = 12                     # Difference threshold
    
    for i in range(num_runs):
        # Test block_sparse_attention (top_k)
        torch.cuda.synchronize()
        start_time = time.time()
        output_top_k = block_sparse_attention(
            q, k, v, top_k, block_size_M, block_size_N
        )
        
    # Test: Random block index test
    Logger.log("\nTesting block_sparse_attention_random")
    test_block_sparse_attention_random(
        B=B, H=H, N=N, D=D, block_size_M=block_size_M, block_size_N=block_size_N, num_runs=num_runs, device=device
    )