import torch
import triton
import triton.language as tl

@triton.jit
def get_mask_fused_anchor_triton(
    Q, K, V, L_buffer, M_buffer, Acc_buffer,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_lb, stride_lh,
    stride_mb, stride_mh,
    stride_acc_z, stride_acc_h, stride_acc_m, stride_acc_k,
    init_cnt: tl.constexpr,
    local_cnt: tl.constexpr,
    sm_scale: tl.constexpr,
    step: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_SZ: tl.constexpr,
    BLOCK_SZ_M: tl.constexpr,
    BLOCK_SZ_N: tl.constexpr,
    dtype: tl.constexpr
):
    # Determine which block of size (step * BLOCK_SZ) is being processed
    block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    # Range of positions in key blocks
    key_positions = tl.arange(0, BLOCK_SZ_N)

    # Compute base offsets for Q, K/V, and buffer pointers
    batch_idx = batch_head_idx // H
    head_idx = batch_head_idx % H
    q_offset = batch_idx * stride_qz + head_idx * stride_qh
    k_offset = batch_idx * stride_kz + head_idx * stride_kh
    l_offset = batch_idx * stride_lb + head_idx * stride_lh
    m_offset = batch_idx * stride_mb + head_idx * stride_mh
    acc_offset = batch_idx * stride_acc_z + head_idx * stride_acc_h

    # Compute how many tokens to include in the initial and local windows
    init_limit = min(init_cnt * BLOCK_SZ * step, (block_idx + 1) * BLOCK_SZ * step)
    local_start = max((block_idx - local_cnt + 1) * BLOCK_SZ * step, 0)

    # Create block pointers for queries, L, M, and accumulator buffers
    q_ptrs = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(block_idx * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ_M, D),
        order=(1, 0)
    )
    l_ptrs = tl.make_block_ptr(
        base=L_buffer + l_offset,
        shape=(N,),
        strides=(1,),
        offsets=(block_idx * BLOCK_SZ * step,),
        block_shape=(BLOCK_SZ_M,),
        order=(0,)
    )
    m_ptrs = tl.make_block_ptr(
        base=M_buffer + m_offset,
        shape=(N,),
        strides=(1,),
        offsets=(block_idx * BLOCK_SZ * step,),
        block_shape=(BLOCK_SZ_M,),
        order=(0,)
    )
    acc_ptrs = tl.make_block_ptr(
        base=Acc_buffer + acc_offset,
        shape=(N, D),
        strides=(stride_acc_m, stride_acc_k),
        offsets=(block_idx * BLOCK_SZ * step, 0),
        block_shape=(BLOCK_SZ_M, D),
        order=(1, 0)
    )

    # Main loop over sub-blocks of queries within the fused window
    for query_start in range(0, BLOCK_SZ * step, BLOCK_SZ_M):
        # Load and scale queries
        q_block = tl.load(q_ptrs)
        q_block = (q_block * sm_scale * 1.44269504).to(dtype)
        # Initialize running max, log-sum-exp accumulator, and output accumulator
        m_running = tl.full([BLOCK_SZ_M], float('-inf'), dtype=tl.float32)
        l_running = tl.zeros([BLOCK_SZ_M], dtype=tl.float32)
        acc_block = tl.zeros([BLOCK_SZ_M, D], dtype=tl.float32)
        # Position indices for each query in this sub-block
        query_positions = query_start + tl.arange(0, BLOCK_SZ_M)

        # Prepare key/value pointers for the initial window
        k_init_ptrs = tl.make_block_ptr(
            base=K + k_offset,
            shape=(D, N),
            strides=(stride_kd, stride_kn),
            offsets=(0, 0),
            block_shape=(D, BLOCK_SZ_N),
            order=(0, 1)
        )
        v_init_ptrs = tl.make_block_ptr(
            base=V + k_offset,
            shape=(N, D),
            strides=(stride_vn, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_SZ_N, D),
            order=(1, 0)
        )
        # Process the initial global window
        for pos in range(0, init_limit, BLOCK_SZ_N):
            k_block = tl.load(k_init_ptrs).to(dtype)
            scores = tl.dot(q_block, k_block)
            # Apply causal mask: positions after query position get -inf
            mask = (key_positions[None, :] + pos) <= query_positions[:, None]
            scores = scores + tl.where(mask, 0, float('-inf'))
            # Numerically stable softmax updates
            new_max = tl.maximum(m_running, tl.max(scores, 1))
            exp_scores = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(m_running - new_max)
            l_running = l_running * alpha + tl.sum(exp_scores, 1)
            acc_block *= alpha[:, None]
            v_block = tl.load(v_init_ptrs)
            acc_block += tl.dot(exp_scores.to(dtype), v_block)
            m_running = new_max
            k_init_ptrs = tl.advance(k_init_ptrs, (0, BLOCK_SZ_N))
            v_init_ptrs = tl.advance(v_init_ptrs, (BLOCK_SZ_N, 0))

        # Prepare key/value pointers for the local window
        k_local_ptrs = tl.make_block_ptr(
            base=K + k_offset,
            shape=(D, N),
            strides=(stride_kd, stride_kn),
            offsets=(0, local_start),
            block_shape=(D, BLOCK_SZ_N),
            order=(0, 1)
        )
        v_local_ptrs = tl.make_block_ptr(
            base=V + k_offset,
            shape=(N, D),
            strides=(stride_vn, stride_vd),
            offsets=(local_start, 0),
            block_shape=(BLOCK_SZ_N, D),
            order=(1, 0)
        )
        # Process the local window around each query block
        local_end = (block_idx + 1) * BLOCK_SZ * step
        for pos in range(init_limit, local_end, BLOCK_SZ_N):
            k_block = tl.load(k_local_ptrs)
            scores = tl.dot(q_block, k_block)
            mask = (key_positions[None, :] + pos) <= query_positions[:, None]
            scores = scores + tl.where(mask, 0, float('-inf'))
            new_max = tl.maximum(m_running, tl.max(scores, 1))
            exp_scores = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(m_running - new_max)
            l_running = l_running * alpha + tl.sum(exp_scores, 1)
            acc_block *= alpha[:, None]
            v_block = tl.load(v_local_ptrs)
            acc_block += tl.dot(exp_scores.to(dtype), v_block)
            m_running = new_max
            k_local_ptrs = tl.advance(k_local_ptrs, (0, BLOCK_SZ_N))
            v_local_ptrs = tl.advance(v_local_ptrs, (BLOCK_SZ_N, 0))

        # Store computed accumulators, L, and M
        tl.store(acc_ptrs, acc_block.to(Acc_buffer.type.element_ty))
        tl.store(l_ptrs, l_running.to(L_buffer.type.element_ty))
        tl.store(m_ptrs, m_running.to(M_buffer.type.element_ty))

        # Advance pointers to next query sub-block
        q_ptrs = tl.advance(q_ptrs, (BLOCK_SZ_M, 0))
        l_ptrs = tl.advance(l_ptrs, (BLOCK_SZ_M,))
        m_ptrs = tl.advance(m_ptrs, (BLOCK_SZ_M,))
        acc_ptrs = tl.advance(acc_ptrs, (BLOCK_SZ_M, 0))


def anchor_attn_step_1(
    Q, K, V,
    init_cnt, local_cnt,
    block_size, step,
    theta, B, H, N, D, sm_scale
):
    # Allocate fp32 buffers for L, M, and accumulator
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device).contiguous()
    M_buffer = torch.full((B, H, N), float('-inf'), dtype=torch.float32, device=Q.device).contiguous()
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32).contiguous()
    # Launch Triton kernel across batch*head and block indices
    grid = (triton.cdiv(N, block_size * step), B * H, 1)
    get_mask_fused_anchor_triton[grid](
        Q, K, V, L_buffer, M_buffer, Acc_buffer,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        L_buffer.stride(0), L_buffer.stride(1),
        M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),
        init_cnt, local_cnt,
        sm_scale, step,
        B, H, N, D,
        BLOCK_SZ=block_size,
        BLOCK_SZ_M=block_size,
        BLOCK_SZ_N=block_size,
        dtype=tl.bfloat16 if Q.dtype==torch.bfloat16 else tl.float16,
        num_warps=8, num_stages=3
    )
    from .get_mask_test import get_q_mask
    return (*anchor_attn_step_2(Q, K, M_buffer, block_size, step, theta, B, H, N, D, sm_scale), L_buffer, M_buffer, Acc_buffer)


def anchor_attn_step_2(
    Q, K, M_buffer,
    block_size, step,
    theta, B, H, N, D, sm_scale
):
    from .get_mask_test import get_q_mask
    # Return sparse coordinates and counts around each query block
    return get_q_mask(Q, K, M_buffer.unsqueeze(-1), block_size, step, theta, B, H, N, D, sm_scale)

if __name__ == "__main__":
    # Example precision and performance check
    B, H, N, D = 1, 32, 128*1024, 128
    block_size, step, theta = 128, 16, 0
    sm_scale = D**-0.5
    init_cnt, local_cnt = 1, 1
    Q = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    V = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    # Run step 1 then step 2 and measure time
    import time
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.time()
        coords, counts, L, M, Acc = anchor_attn_step_1(Q, K, V, init_cnt, local_cnt, block_size, step, theta, B, H, N, D, sm_scale)
        torch.cuda.synchronize()
        print(f"Fused mask compute time: {(time.time()-start)*1000:.2f} ms")
