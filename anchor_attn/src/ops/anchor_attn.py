import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import triton
import triton.language as tl
import time
from my_utils import Logger


@triton.jit
def _triton_block_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_index, block_index_context,
    Out, L_buffer, M_buffer, Acc_buffer,  # Accumulator buffer for outputs
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_mb, stride_mh,
    stride_acc_z, stride_acc_h, stride_acc_m, stride_acc_k,
    stride_bic_z, stride_bic_h, stride_bic_m,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    # Identify row and batch-head
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Load sequence length for this batch
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # Compute index offsets for queries, keys, and dimensions
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute memory offsets for each tensor
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    lm_offset = (off_hz // H) * stride_lb + (off_hz % H) * stride_lh
    mm_offset = (off_hz // H) * stride_mb + (off_hz % H) * stride_mh
    acc_offset = (off_hz // H) * stride_acc_z + (off_hz % H) * stride_acc_h

    # Build pointers into Q, K, V, output, and buffers
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    l_ptrs = L_buffer + lm_offset + offs_m
    m_ptrs = M_buffer + mm_offset + offs_m
    acc_ptrs = Acc_buffer + acc_offset + offs_m[:, None] * stride_acc_m + offs_d[None, :] * stride_acc_k
    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW
    bic_ptr = block_index_context + (off_hz // H) * stride_bic_z + (off_hz % H) * stride_bic_h + start_m * stride_bic_m

    # Initialize softmax accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Pre-scale Q for dot-product
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # Mask invalid rows
    valid_mask = offs_m[:, None] < seqlen

    # Load number of valid sparse blocks
    block_count = tl.minimum(tl.load(bic_ptr), MAX_BLOCKS_PRE_ROW)

    # Process each sparse block
    for idx in range(block_count):
        blk = tl.load(blocks_ptr + idx)
        start_n = blk * BLOCK_N
        cols = start_n + offs_n

        # Load K and V for this block
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)

        # Compute masked dot-product QK
        qk = tl.dot(q, k)
        causal = cols[None, :] <= offs_m[:, None]
        qk = tl.where(valid_mask & causal, qk, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Store intermediate accumulator
    tl.store(acc_ptrs, acc.to(dtype), mask=valid_mask)
    # Final normalize
    out = acc / l_i[:, None]
    tl.store(o_ptrs, out.to(dtype), mask=valid_mask)
    # Save updated l_i and m_i
    tl.store(l_ptrs, l_i, mask=offs_m < seqlen)
    tl.store(m_ptrs, m_i, mask=offs_m < seqlen)


@triton.jit
def _triton_block_sparse_attn_fwd_kernel_step2(
    Q, K, V, seqlens, sm_scale,
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
    Z, H, N,
    NUM_ROWS_STEP2, MAX_INDICES,
    STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_INDICES: tl.constexpr,
    dtype: tl.constexpr,
):
    # Get program indices
    start_row = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    query_start = start_row * STEP
    if query_start >= seqlen:
        return

    # Offsets for rows and dims
    offs_m = query_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Base memory offsets for Q, output, buffers
    qo_off = (off_hz//H)*stride_qz + (off_hz%H)*stride_qh
    kv_off = (off_hz//H)*stride_kz + (off_hz%H)*stride_kh
    lb_off = (off_hz//H)*stride_lb + (off_hz%H)*stride_lh
    ab_off = (off_hz//H)*stride_abz + (off_hz%H)*stride_abh
    tc_off = (off_hz//H)*stride_tcz + (off_hz%H)*stride_tch + start_row*stride_tcm
    tct_off = (off_hz//H)*stride_tctz + (off_hz%H)*stride_tcth + start_row*stride_tctm

    # Pointers for Q, K, V, output, and buffers
    q_ptr = Q + qo_off + offs_m[:,None]*stride_qm + offs_d[None,:]*stride_qk
    k_ptr = K + kv_off + offs_d[:,None]*stride_kk
    v_ptr = V + kv_off + offs_d[None,:]*stride_vk
    o_ptr = Out + qo_off + offs_m[:,None]*stride_om + offs_d[None,:]*stride_ok
    l_ptr = L_buffer + lb_off + offs_m
    m_ptr = M_buffer + lb_off + offs_m
    acc_ptr = Acc_buffer + ab_off + offs_m[:,None]*stride_abm + offs_d[None,:]*stride_abd
    coord_ptr = true_coords + tc_off
    count_ptr = true_counts + tct_off

    # Load initial l, m, accumulator
    m_i = tl.load(m_ptr, mask=offs_m<seqlen, other=-float('inf'))
    l_i = tl.load(l_ptr, mask=offs_m<seqlen, other=0.0)
    acc = tl.load(acc_ptr, mask=offs_m[:,None]<seqlen, other=0.0)

    # Scale factor
    scale = sm_scale * 1.44269504
    sparse_count = tl.load(count_ptr)

    # Process each index batch
    for start_idx in range(0, sparse_count, NUM_INDICES):
        idxs = start_idx + tl.arange(0, NUM_INDICES)
        valid_idx = idxs < sparse_count
        real_blk = tl.load(coord_ptr + idxs, mask=valid_idx, other=0)
        offs_n = real_blk
        key_valid = offs_n < seqlen
        mask = valid_idx & key_valid

        # Load Q block
        q = tl.load(q_ptr, mask=offs_m[:,None]<seqlen, other=0.0)
        q = (q * scale).to(dtype)

        # Load K and V groups
        k = tl.load(k_ptr + offs_n[None,:]*stride_kn, mask=mask[None,:], other=0.0)
        v = tl.load(v_ptr + offs_n[:,None]*stride_vn, mask=mask[:,None], other=0.0)

        # Compute dot-product
        qk = tl.dot(q, k)
        qk = tl.where(mask[None,:], qk, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk,1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:,None])
        acc = acc * alpha[:,None] + tl.dot(p.to(dtype), v)
        l_i = l_i*alpha + tl.sum(p,1)
        m_i = m_new

    # Final normalization and store
    acc = acc / l_i[:,None]
    tl.store(o_ptr, acc.to(dtype), mask=offs_m[:,None]<seqlen)


@triton.jit
def attention_kernel(
    Q, K, V, Output, L_buffer, M_buffer,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_mb, stride_mh,
    B, H, N, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    qk_scale: tl.constexpr
):
    # Compute program IDs for sub-blocks
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_z, off_h = divmod(off_hz, H)

    # Compute base offsets
    q_off = off_z*stride_qz + off_h*stride_qh
    k_off = off_z*stride_kz + off_h*stride_kh
    v_off = off_z*stride_vz + off_h*stride_vh
    o_off = off_z*stride_oz + off_h*stride_oh
    l_off = off_z*stride_lb + off_h*stride_lh
    m_off = off_z*stride_mb + off_h*stride_mh

    # Make Triton block pointers for Q, K, V, and output
    Q_block = tl.make_block_ptr(Q+q_off, (N,D), (stride_qm,stride_qd), (start_m,0), (BLOCK_M,D), (1,0))
    K_block = tl.make_block_ptr(K+k_off, (D,N), (stride_kd,stride_kn), (0,0), (D,BLOCK_N), (0,1))
    V_block = tl.make_block_ptr(V+v_off, (N,D), (stride_vn,stride_vd), (0,0), (BLOCK_N,D), (1,0))
    O_block = tl.make_block_ptr(Output+o_off, (N,D), (stride_om,stride_od), (start_m,0), (BLOCK_M,D), (1,0))

    # Load Q sub-block
    q = tl.load(Q_block, boundary_check=(0,1))
    # Load previous softmax state
    offs_m = start_m + tl.arange(0,BLOCK_M)
    valid_m = offs_m < N
    l_i = tl.load(L_buffer + l_off + offs_m, mask=valid_m)
    m_i = tl.load(M_buffer + m_off + offs_m, mask=valid_m)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Full attention over all key blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < N
        mask = (offs_n[None,:] <= offs_m[:,None]) & valid_n[None,:]

        # Load K and V blocks
        k = tl.load(K_block, boundary_check=(0,1))
        v = tl.load(V_block, boundary_check=(0,1))

        # Compute QK and apply mask/scale
        qk = tl.dot(q, k) * qk_scale
        qk = tl.where(mask, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        scale_factor = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:,None])
        l_batch = tl.sum(p, axis=1)

        acc = acc*scale_factor[:,None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i*scale_factor + l_batch
        m_i = m_new

        # Advance K/V pointers
        K_block = tl.advance(K_block, (0, BLOCK_N))
        V_block = tl.advance(V_block, (BLOCK_N, 0))

    # Normalize and store output
    out = acc / l_i[:,None]
    tl.store(O_block, out.to(Output.type.element_ty), boundary_check=(0,1))
    tl.store(L_buffer + l_off + offs_m, l_i, mask=valid_m)
    tl.store(M_buffer + m_off + offs_m, m_i, mask=valid_m)


def get_block_idx(B, H, NUM_ROWS, step, device):
    # Generate block indices for sparse attention
    block_index = torch.zeros((B, H, NUM_ROWS, step+2), dtype=torch.int32, device=device)
    block_index_context = torch.zeros((B, H, NUM_ROWS, 1), dtype=torch.int32, device=device)
    for i in range(0, NUM_ROWS, step):
        start = i
        end = min(i+step, NUM_ROWS)
        idx = torch.arange(start, end, device=device)
        repeat = idx.repeat(end-start,1)
        thresholds = torch.arange(start, end, device=device)[:,None]
        repeat = torch.where(repeat>thresholds, 0, repeat)
        count = idx - start + 1
        if i>0:
            block_index[..., start:end,2:repeat.size(-1)+2] = repeat[None,None]
            block_index[..., start:end,0] = 0
            block_index[..., start:end,1] = i-1
        else:
            block_index[..., start:end,:repeat.size(-1)] = repeat[None,None]
        block_index_context[..., start:end, 0] = count + (2 if i>0 else 0)
    return block_index, block_index_context



def _anchor_attn(Q, K, V,
                        sm_scale: float= None,
                        block_size:int=32,
                        block_size_N:int=32,
                        step:int=16,
                        difference:int=100,
                        return_sparsity_ratio: bool = False,
                        ):
    B, H, N, D = Q.shape
    assert N % block_size == 0
    NUM_ROWS = N // block_size
    
    # 不计时间可以静态设置
    block_index, block_index_context = get_block_idx(B,H,NUM_ROWS,step,Q.device)
    # block_index_context = block_index_context.contiguous()
    # block_index = block_index.contiguous()
    seqlens = torch.full((B,), N , dtype=torch.int32, device=Q.device)
    # Logger.log(f"block_index shape:{block_index.shape}")
    # Logger.log(f"block_index_context shape:{block_index_context.shape}")

    NUM_ROWS_STEP2 = (N // block_size + step - 1) // step
    MAX_BLOCKS_PRE_ROW = block_index.shape[-1]
    output = torch.zeros_like(Q, device=Q.device)
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device)
    M_buffer = torch.full((B, H, N), -float("inf"), dtype=torch.float32, device=Q.device)
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32, device=Q.device)

    torch.cuda.synchronize()
    # step 1 
    get_anchor = time.time()
    grid = (NUM_ROWS, B * H)
    _triton_block_sparse_attn_fwd_kernel[grid](
        Q, K, V, seqlens, sm_scale,
        block_index, block_index_context,
        output, L_buffer, M_buffer,Acc_buffer,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),
        block_index_context.stride(0), block_index_context.stride(1), block_index_context.stride(2),
        B, H, N,
        NUM_ROWS, MAX_BLOCKS_PRE_ROW,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        BLOCK_DMODEL=D,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    )
    torch.cuda.synchronize()
    Logger.log(f"get anchor cost:{(time.time() - get_anchor) * 1000}ms")
    
    from .get_mask_test import get_q_mask
    get_mask = time.time()
    true_coords, true_counts = get_q_mask(Q,K,M_buffer.unsqueeze(-1), 
                                          block_size,step,difference,
                                          B, H, N, D, 
                                          block_size,
                                          sm_scale,
                                          )
    torch.cuda.synchronize()
    true_coords = true_coords.contiguous()
    true_counts = true_counts.contiguous()
    Logger.log(f"get mask cost:{(time.time() - get_mask) * 1000}ms")
 
    MAX_BLOCKS_PRE_ROW_STEP2 = true_coords.shape[-1]
    
    # Second kernel: Update with sparse key blocks
    grid_step2 = (NUM_ROWS_STEP2, B * H)
    torch.cuda.synchronize()
    sparse_time = time.time()
    
    # Logger.log(f"true_coords.stride()")
    # Logger.log(f"true_coords.stride()")
    Logger.log(f"true_coords:{true_coords.shape}")
    Logger.log(f"true_counts:{true_counts.shape}")
    # true_counts = torch.empty((B, H, N // block_size // step, 1), dtype=torch.int32, device=Q.device).contiguous()
    sparsity_ratio = true_counts.sum() / (B * H * N / (block_size * step)* N // 2)
    # Logger.log(f"anchor sparsity_ratio:{sparsity_ratio}")
    # debug_offs_n = torch.zeros_like(true_coords,dtype=torch.int32)
    # true_coords[0,0,:,:] = torch.sort(true_coords[0,0,:,:],dim=-1,descending=True)[0]
    # Logger.log(f"true_coords:{true_coords[0,0,:,:10].tolist()}")
    # Logger.log(f"true_counts:{true_counts[0,0,:,:10].tolist()}")
    _triton_block_sparse_attn_fwd_kernel_step2[grid_step2](
        Q, K, V, seqlens, sm_scale,
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
        NUM_ROWS_STEP2, 
        MAX_BLOCKS_PRE_ROW_STEP2,
        STEP = step * block_size, 
        BLOCK_M = 128, # config
        BLOCK_N = 32,
        BLOCK_DMODEL=D,
        NUM_INDICES=128,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        # debug_offs_n=debug_offs_n,
        # stride_dnz=debug_offs_n.stride(0),
        # stride_dnh=debug_offs_n.stride(1),
        # stride_dnm=debug_offs_n.stride(2),
        # stride_dnk=debug_offs_n.stride(3),
        num_warps=8,
        num_stages=3
    )
    
    torch.cuda.synchronize()
    Logger.log(f"sparse cost:{(time.time() - sparse_time) * 1000}ms")
    if return_sparsity_ratio:
        sparsity_ratio = true_counts.sum() / (B * H * N / (block_size * step)* N// 2)
        return output,sparsity_ratio
    sparsity_ratio = true_counts.sum() / (B * H * N / (block_size * step)* N// 2)
    Logger.log(f"anchor sparsity_ratio:{sparsity_ratio}")
    return output,1.0

def anchor_attn(Q, K, V,
                           sm_scale:float = None,
                           block_size_M: int = 32,
                           block_size_N: int = 32,
                           step: int = 16,
                           difference:int = 15,
                           return_computational_ratio: bool = True
                           ):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert Q.size(-2) % block_size_M == 0
    B, H, N, D = Q.shape
    sm_scale = sm_scale if sm_scale else 1.0 / (D ** 0.5)
    sparse_output,sparse_ratio = _anchor_attn(Q, K, V, 
                                            sm_scale, 
                                            block_size_M,
                                            block_size_N,
                                            step,
                                            difference,
                                            return_computational_ratio,
                                        )
    if return_computational_ratio: 
        return  sparse_output,sparse_ratio
    return sparse_output

# Test function (updated to include a basic check)
def test_window_attention(B=1, H=32, N=32*4096+32, D=128, step=16, num_runs=3):
    import time
    Logger.set_log_file_path(f"log/anchor_test/{time.time()}.log")
    torch.manual_seed(42)
    device = "cuda:0"
    Q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device).contiguous() * 2
    K = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device).contiguous() * 2
    V = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device).contiguous() / 2
    
    import time
    for i in range(3):
        torch.cuda.synchronize()  # Ensure GPU is ready
        start_time = time.time()
        flash_attn=_flash_attention_forward(
                Q.transpose(1,2),
                K.transpose(1,2),
                V.transpose(1,2),
                attention_mask = None,
                query_length = N ,
                is_causal = True,
            ).transpose(1,2)
        
        torch.cuda.synchronize()  # Ensure GPU is ready
        elapsed_time = (time.time() - start_time) * 1000
        Logger.log(f"flash attn cost:{elapsed_time}ms")
    Logger.log(f"flash_attn shape:{flash_attn.shape}")
    
    for i in range(4):
        torch.cuda.synchronize()  # Ensure GPU is ready
        start_time = time.time()
        anchor_attn_out,sparse_ratio = anchor_attn(Q,K,V)    
        torch.cuda.synchronize()  # Ensure GPU is ready
        elapsed_time = (time.time() - start_time) * 1000
        Logger.log(f"anchor block attn cost:{elapsed_time}ms")
        Logger.log(f"sparse ratio:{sparse_ratio}")
    Logger.log(f"anchor_attn shape:{anchor_attn_out.shape}")
    from my_utils.ops_util import get_deff
    get_deff(flash_attn,anchor_attn_out)


if __name__ == "__main__":
    test_window_attention(B=1, H=32, N=1024*128, D=128, step=16, num_runs=3)
