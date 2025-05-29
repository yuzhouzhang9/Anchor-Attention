import torch
import triton
import triton.language as tl
import time
from my_utils import Logger

@triton.jit
def _triton_stripe_sparse_attn_fwd_kernel_step3(
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
    # Determine program indices for major dimensions
    start_m = tl.program_id(0)
    big_block_startm = (start_m * BLOCK_M) // STEP
    off_hz = tl.program_id(1)

    # Compute offsets for query, key/value, and buffers
    query_start = start_m * BLOCK_M
    offs_m = query_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    lm_offset = (off_hz // H) * stride_lb + (off_hz % H) * stride_lh
    ab_offset = (off_hz // H) * stride_abz + (off_hz % H) * stride_abh

    # Offsets for coordinate and count buffers
    coord_offset = (off_hz // H) * stride_tcz + (off_hz % H) * stride_tch + big_block_startm * stride_tcm
    count_offset = (off_hz // H) * stride_tctz + (off_hz % H) * stride_tcth + big_block_startm * stride_tctm

    # Prepare pointers for loading keys and values
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk

    # Load sparsity count for this block
    count_ptr = true_counts + count_offset
    sparse_count = tl.load(count_ptr)

    # Reset query offsets
    offs_m = query_start + tl.arange(0, BLOCK_M)
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    l_ptrs = L_buffer + lm_offset + offs_m
    m_ptrs = M_buffer + lm_offset + offs_m
    a_ptrs = Acc_buffer + ab_offset + offs_m[:, None] * stride_abm + offs_d[None, :] * stride_abd

    # Load initial buffers and scaled queries
    m_i = tl.load(m_ptrs)
    l_i = tl.load(l_ptrs)
    acc = tl.load(a_ptrs)
    q = tl.load(q_ptrs)
    q = (q * sm_scale * 1.44269504).to(dtype)  # Convert to log2 scale

    # Load true coordinate pointers for iteration
    coord_ptr = true_coords + coord_offset

    # Iterate through the selected sparse indices in chunks
    for idx_start in range(0, sparse_count, NUM_INDICES):
        idx_range = idx_start + tl.arange(0, NUM_INDICES)
        valid_mask = idx_range < sparse_count
        block_ids = tl.load(coord_ptr + idx_range, mask=valid_mask, other=0)

        # Compute pointers within the window
        k_window = k_ptrs + block_ids[None, :] * stride_kn
        v_window = v_ptrs + block_ids[:, None] * stride_vn

        # Load masked keys and values
        k = tl.load(k_window, mask=valid_mask[None, :], other=0.0)
        v = tl.load(v_window, mask=valid_mask[:, None], other=0.0)

        # Compute QK^T for this mini-batch
        qk = tl.zeros([BLOCK_M, NUM_INDICES], dtype=dtype)
        qk += tl.dot(q, k)
        qk = tl.where(valid_mask[None, :], qk, float("-inf"))

        # Update running max for numerical stability
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        # Scale accumulator and add new weighted values
        acc *= alpha[:, None]
        acc += tl.dot(p.to(dtype), v)

        # Update L and M buffers
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Finalize attention output
    result = acc / l_i[:, None]
    tl.store(o_ptrs, result.to(dtype))

def _anchor_attn_test_time(Q, K, V,
                        sm_scale: float= None,
                        block_size:int=64, 
                        step:int=16,
                        difference:float=100,
                        return_sparsity_ratio: bool = False,
                        BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128):
    B, H, N, D = Q.shape
    assert N % block_size == 0
    NUM_ROWS_STEP3 = triton.cdiv(N, block_size * step)
    output = torch.zeros_like(Q, device=Q.device)
    
      
    from .anchor_step_1_2_v2 import anchor_attn_step_1, anchor_attn_step_2
    torch.cuda.synchronize()
    start_time = time.time()
    L_buffer, M_buffer, Acc_buffer = anchor_attn_step_1(Q, K, V, 
                                                        1, 1,
                                                        block_size, 
                                                        step,  
                                                        B, H, N, D, sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N)
    true_coords, true_counts = anchor_attn_step_2(Q, K, M_buffer,
                                                    block_size, 
                                                    step, difference, 
                                                    B, H, N, D, sm_scale) 
    torch.cuda.synchronize()
    end_time = time.time()
    search_time = (end_time - start_time) * 1000


    grid_step3 = (triton.cdiv(N, BLOCK_SIZE_M), B * H) 
    
    torch.cuda.synchronize()
    start_time = time.time()
    _triton_stripe_sparse_attn_fwd_kernel_step3[grid_step3](
        Q, K, V, sm_scale,
        true_coords, true_counts,
        output, L_buffer, M_buffer, Acc_buffer,  # 传递 Acc_buffer
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),  # Acc_buffer 步幅
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, N,
        STEP = step * block_size, 
        BLOCK_M = BLOCK_SIZE_M, # config
        BLOCK_DMODEL=D,
        NUM_INDICES=128,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    ) 
    torch.cuda.synchronize()
    end_time = time.time()
    compute_time = (end_time - start_time) * 1000
    if return_sparsity_ratio:
        sparsity_ratio = true_counts.sum() / (B * H * N * N / (block_size * step)  // 2)
        return output,sparsity_ratio, search_time, compute_time
    return output,1.0, search_time, compute_time

def _anchor_attn_without_anchor(Q, K, V, 
                        sm_scale: float= None,
                        block_size:int=64, 
                        step:int=16,
                        difference:float=100, 
                        BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128):
    B, H, N, D = Q.shape
    assert N % block_size == 0
    NUM_ROWS_STEP3 = triton.cdiv(N, block_size * step)
    output = torch.zeros_like(Q, device=Q.device)
    
      
    from .anchor_step_1_2_v2 import anchor_attn_step_1, anchor_attn_step_2
    L_buffer, M_buffer, Acc_buffer = anchor_attn_step_1(Q, K, V, 
                                                        1, 1,
                                                        block_size, 
                                                        step,  
                                                        B, H, N, D, 
                                                        sm_scale, 
                                                        BLOCK_SIZE_M, 
                                                        BLOCK_SIZE_N)
    
    true_coords, true_counts = anchor_attn_step_2(Q, K, M_buffer,
                                                    block_size, 
                                                    step, difference, 
                                                    B, H, N, D, 
                                                    sm_scale)  
    
    
    grid_step3 = (triton.cdiv(N, BLOCK_SIZE_M), B * H) 
    _triton_stripe_sparse_attn_fwd_kernel_step3[grid_step3](
        Q, K, V, sm_scale,
        true_coords, true_counts,
        output, L_buffer, M_buffer, Acc_buffer,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),  # Acc_buffer 步幅
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, N,
        STEP = step * block_size, 
        BLOCK_M = BLOCK_SIZE_M, # config
        BLOCK_DMODEL=D,
        NUM_INDICES=128,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    ) 
    sparsity_ratio = true_counts.sum() / (B * H * N * N / (block_size * step)  // 2)
    return output,sparsity_ratio 

def _anchor_attn(Q, K, V,
                    sm_scale: float= None,
                    block_size:int=64, 
                    step:int=16,
                    difference:float=100,
                    return_sparsity_ratio: bool = False,
                    BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128):
    B, H, N, D = Q.shape
    assert N % block_size == 0
    output = torch.zeros_like(Q, device=Q.device)
    from .anchor_step_1_2_v2 import anchor_attn_step_1, anchor_attn_step_2
    L_buffer, M_buffer, Acc_buffer = anchor_attn_step_1(Q, K, V, 
                                                        1, 1,
                                                        block_size, 
                                                        step,  
                                                        B, H, N, D, sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N)
    true_coords, true_counts = anchor_attn_step_2(Q, K, M_buffer,
                                                    block_size, 
                                                    step, difference, 
                                                    B, H, N, D, sm_scale) 
    grid_step3 = (triton.cdiv(N, BLOCK_SIZE_M), B * H) 
    _triton_stripe_sparse_attn_fwd_kernel_step3[grid_step3](
        Q, K, V, sm_scale,
        true_coords, true_counts,
        output, L_buffer, M_buffer, Acc_buffer,  # 传递 Acc_buffer
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),  # Acc_buffer 步幅
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        B, H, N,
        STEP = step * block_size, 
        BLOCK_M = BLOCK_SIZE_M, # config
        BLOCK_DMODEL=D,
        NUM_INDICES=128,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    ) 
    
    if return_sparsity_ratio:
        sparsity_ratio = true_counts.sum() / (B * H * N * N / (block_size * step)  // 2)
        return output,sparsity_ratio
    return output,1.0

def anchor_attn_without_anchor(Q, K, V,
                        sm_scale:float = None,
                        block_size: int = 128, 
                        step: int = 16,
                        difference:float = 12 
                           ):
    difference *= 1.44269504
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert Q.size(-2) % block_size == 0
    B, H, N, D = Q.shape
    sm_scale = sm_scale if sm_scale else 1.0 / (D ** 0.5)
    return _anchor_attn_without_anchor(Q, K, V,sm_scale,block_size,step,difference) 

def anchor_attn(Q, K, V,
                        sm_scale:float = None,
                        block_size: int = 128, 
                        step: int = 16,
                        difference:float = 12,
                        return_computational_ratio: bool = True
                           ):
    difference *= 1.44269504
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), \
        "Q, K, V must be contiguous. Please call .contiguous() before passing."

    assert step in [16, 32], \
        f"Invalid step size: {step}. Only step sizes 16 or 32 are supported."

    assert Q.size(-2) % (block_size * step) == 0, \
        f"Sequence length {Q.size(-2)} must be divisible by (block_size * step = {block_size * step})."

    B, H, N, D = Q.shape
    sm_scale = sm_scale if sm_scale else 1.0 / (D ** 0.5)
    sparse_output,sparse_ratio = _anchor_attn(Q, K, V, 
                                            sm_scale, 
                                            block_size, 
                                            step,
                                            difference,
                                            return_computational_ratio,
                                        )
    if return_computational_ratio: 
        from my_utils import Logger
        Logger.log(f"anchor sparse_ratio:{sparse_ratio}")
        return sparse_output,sparse_ratio
    return sparse_output


# the function is same as anchor_attn, but with return computational time, which is used the sync function 
def anchor_attn_test_time(Q, K, V,
                        sm_scale:float = None,
                        block_size: int = 128, 
                        step: int = 16,
                        difference:float = 12,
                        return_computational_ratio: bool = True
                           ):
    difference *= 1.44269504
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert Q.size(-2) % block_size == 0
    B, H, N, D = Q.shape
    sm_scale = sm_scale if sm_scale else 1.0 / (D ** 0.5)
    sparse_output,sparse_ratio,search_time,compute_time = _anchor_attn_test_time(Q, K, V, 
                                            sm_scale, 
                                            block_size, 
                                            step,
                                            difference,
                                            return_computational_ratio,
                                        )
    if return_computational_ratio: 
        from my_utils import Logger
        Logger.log(f"anchor sparse_ratio:{sparse_ratio}")
        return  sparse_output,sparse_ratio,search_time,compute_time
    return sparse_output,search_time,compute_time



# Test function (updated to include a basic check)
def test_window_attention(B=1, H=32, N=12*1024, D=128):
    import time
    from my_utils import Logger
    Logger.set_log_file_path(f"log/anchor_test_v2/{time.time()}.log")
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda:0").contiguous() * 2
    K = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda:0").contiguous() * 2
    V = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda:0").contiguous() / 2
    
    import time
    for i in range(4):
        torch.cuda.synchronize()  # Ensure GPU is ready
        start_time = time.time()
        anchor_attn_out,sparse_ratio = anchor_attn(Q,K,V)    
        torch.cuda.synchronize()  # Ensure GPU is ready
        elapsed_time = (time.time() - start_time) * 1000
        Logger.log(f"anchor attn cost:{elapsed_time}ms")



if __name__ == "__main__":
    test_window_attention(B=1, H=1, N=1024*512, D=128)