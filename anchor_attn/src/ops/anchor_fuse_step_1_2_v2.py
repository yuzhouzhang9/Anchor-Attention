import torch
import triton
import triton.language as tl
import time
import os
from my_utils import Logger

@triton.jit
def get_mask_fused_anchor_triton(
    Q, K, V, L_buffer, M_buffer, Acc_buffer, true_coords, true_counts,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_lb, stride_lh, stride_mb, stride_mh,
    stride_acc_z, stride_acc_h, stride_acc_m, stride_acc_k,
    stride_cz, stride_ch, stride_cn, stride_cd,
    stride_tcz, stride_tch, stride_tcn, stride_tcd,
    init_cnt: tl.constexpr,
    local_cnt: tl.constexpr,
    sm_scale: tl.constexpr,
    step: tl.constexpr,
    theta: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_SZ: tl.constexpr,
    BLOCK_SZ_A: tl.constexpr,
    BLOCK_SZ_M: tl.constexpr,
    BLOCK_SZ_N: tl.constexpr,
    dtype: tl.constexpr
):
    # Determine current block and batch-head indices
    block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    # Precompute key positions within a key block
    key_positions = tl.arange(0, BLOCK_SZ_N)

    # Calculate strides for Q, K/V, L, M, and Acc buffers
    batch_idx = batch_head_idx // H
    head_idx = batch_head_idx % H
    q_offset = batch_idx * stride_qz + head_idx * stride_qh
    k_offset = batch_idx * stride_kz + head_idx * stride_kh
    l_offset = batch_idx * stride_lb + head_idx * stride_lh
    m_offset = batch_idx * stride_mb + head_idx * stride_mh
    acc_offset = batch_idx * stride_acc_z + head_idx * stride_acc_h

    # Define global token ranges for initial and local windows
    init_limit = min(init_cnt * BLOCK_SZ * step, (block_idx + 1) * BLOCK_SZ * step)
    local_start = max((block_idx - local_cnt + 1) * BLOCK_SZ * step, 0)

    # Create Triton block pointers for Query, L, M, and Acc buffers
    q_ptrs = tl.make_block_ptr(base=Q + q_offset, shape=(N, D), strides=(stride_qm, stride_qd), offsets=(block_idx * BLOCK_SZ * step, 0), block_shape=(BLOCK_SZ_M, D), order=(1, 0))
    l_ptrs = tl.make_block_ptr(base=L_buffer + l_offset, shape=(N,), strides=(1,), offsets=(block_idx * BLOCK_SZ * step,), block_shape=(BLOCK_SZ_M,), order=(0,))
    m_ptrs = tl.make_block_ptr(base=M_buffer + m_offset, shape=(N,), strides=(1,), offsets=(block_idx * BLOCK_SZ * step,), block_shape=(BLOCK_SZ_M,), order=(0,))
    acc_ptrs = tl.make_block_ptr(base=Acc_buffer + acc_offset, shape=(N, D), strides=(stride_acc_m, stride_acc_k), offsets=(block_idx * BLOCK_SZ * step, 0), block_shape=(BLOCK_SZ_M, D), order=(1, 0))

    # Precompute scaling factor for dot-products in log2 domain
    qk_scale = sm_scale * 1.44269504

    # Initialize containers for per-window maxima and average queries
    m_total = tl.full([BLOCK_SZ * step], float('-inf'), dtype=tl.float32)
    q_avg_pool = tl.zeros([step, D], dtype=dtype)

    # Process each sub-block of queries within the fused window
    for i in range(0, BLOCK_SZ * step, BLOCK_SZ_M):
        # Compute positions for this query sub-block
        base_pos = block_idx * BLOCK_SZ * step
        query_positions = tl.arange(0, BLOCK_SZ_M) + i + base_pos
        # Load and scale queries
        q_block = tl.load(q_ptrs)
        q_block = (q_block * qk_scale).to(dtype)

        # Compute block-wise average queries for anchor selection
        group_size = BLOCK_SZ_M // BLOCK_SZ
        q_avg = tl.sum(q_block.reshape([group_size, BLOCK_SZ, D]), axis=1) / BLOCK_SZ
        # Expand the averaged queries across the 'step' dimension
        repeats = step // group_size
        expanded = tl.broadcast_to(q_avg[None, :, :], (repeats, group_size, D)).reshape([step, D])
        # Determine indices where to place averages
        indices = (tl.arange(0, step) % group_size) + (i // BLOCK_SZ)
        q_avg_pool = tl.where((tl.arange(0, step)[:, None] == indices[:, None]), expanded.to(dtype), q_avg_pool)

        # Initialize softmax accumulators for this sub-block
        m_running = tl.full([BLOCK_SZ_M], float('-inf'), dtype=tl.float32)
        l_running = tl.zeros([BLOCK_SZ_M], dtype=tl.float32)
        acc_block = tl.zeros([BLOCK_SZ_M, D], dtype=tl.float32)

        # Prepare key/value pointers for initial and local windows
        k_init_ptrs = tl.make_block_ptr(base=K + k_offset, shape=(D, N), strides=(stride_kd, stride_kn), offsets=(0, 0), block_shape=(D, BLOCK_SZ_N), order=(0, 1))
        v_init_ptrs = tl.make_block_ptr(base=V + k_offset, shape=(N, D), strides=(stride_vn, stride_vd), offsets=(0, 0), block_shape=(BLOCK_SZ_N, D), order=(1, 0))
        k_local_ptrs = tl.make_block_ptr(base=K + k_offset, shape=(D, N), strides=(stride_kd, stride_kn), offsets=(0, local_start), block_shape=(D, BLOCK_SZ_N), order=(0, 1))
        v_local_ptrs = tl.make_block_ptr(base=V + k_offset, shape=(N, D), strides=(stride_vn, stride_vd), offsets=(local_start, 0), block_shape=(BLOCK_SZ_N, D), order=(1, 0))

        # Compute attention over initial window tokens
        for pos in range(0, init_limit, BLOCK_SZ_N):
            k_block = tl.load(k_init_ptrs).to(dtype)
            scores = tl.dot(q_block, k_block)
            mask = (key_positions[None, :] + pos) <= query_positions[:, None]
            scores = scores + tl.where(mask, 0, float('-inf'))
            new_max = tl.maximum(m_running, tl.max(scores, 1))
            exp_scores = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(m_running - new_max)
            l_running = l_running * alpha + tl.sum(exp_scores, 1)
            acc_block *= alpha[:, None]
            v_block = tl.load(v_init_ptrs)
            acc_block += tl.dot(exp_scores.to(dtype), v_block)
            m_running = new_max
            # Record per-query maxima in m_total
            m_total = tl.where(tl.arange(0, BLOCK_SZ * step) == (i + tl.arange(0, BLOCK_SZ_M)), tl.broadcast_to(new_max[None, :], [BLOCK_SZ * step]), m_total)
            k_init_ptrs = tl.advance(k_init_ptrs, (0, BLOCK_SZ_N))
            v_init_ptrs = tl.advance(v_init_ptrs, (BLOCK_SZ_N, 0))

        # Compute attention over local window tokens
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

        # Store updated buffers for this sub-block
        tl.store(Acc_buffer + acc_offset + (block_idx * BLOCK_SZ * step + i) * stride_acc_m, acc_block.to(Acc_buffer.type.element_ty))
        tl.store(L_buffer + l_offset + (block_idx * BLOCK_SZ * step + i) * stride_lb, l_running.to(L_buffer.type.element_ty))
        tl.store(M_buffer + m_offset + (block_idx * BLOCK_SZ * step + i) * stride_mb, m_running.to(M_buffer.type.element_ty))

        # Advance pointers to the next query sub-block
        q_ptrs = tl.advance(q_ptrs, (BLOCK_SZ_M, 0))
        l_ptrs = tl.advance(l_ptrs, (BLOCK_SZ_M,))
        m_ptrs = tl.advance(m_ptrs, (BLOCK_SZ_M,))
        acc_ptrs = tl.advance(acc_ptrs, (BLOCK_SZ_M, 0))

    # After processing queries, select anchor tokens based on m_total and averaged queries
    coords_offset = batch_idx * stride_cz + head_idx * stride_ch
    counts_offset = batch_idx * stride_tcz + head_idx * stride_tch
    coords_ptr = tl.make_block_ptr(base=true_coords + coords_offset, shape=(N // BLOCK_SZ // step, N), strides=(stride_cn, stride_cd), offsets=(block_idx, 0), block_shape=(1, BLOCK_SZ_N), order=(1, 0))
    counts_ptr = tl.make_block_ptr(base=true_counts + counts_offset, shape=(N // BLOCK_SZ // step, 1), strides=(stride_tcn, stride_tcd), offsets=(block_idx, 0), block_shape=(1, 1), order=(1, 0))
    k_select_ptrs = tl.make_block_ptr(base=K + k_offset, shape=(D, N), strides=(stride_kd, stride_kn), offsets=(0, init_limit), block_shape=(D, BLOCK_SZ_N), order=(0, 1))

    # Compute anchor score pools by averaging maxima per step
    anchor_pool = tl.sum(m_total.reshape([step, BLOCK_SZ]), axis=1) / BLOCK_SZ
    count = tl.zeros([1, 1], dtype=tl.int32)

    # Select anchors where query_avg·key ≥ anchor threshold
    for pos in range(init_limit, local_start, BLOCK_SZ_N):
        k_block = tl.load(k_select_ptrs)
        scores = tl.dot(q_avg_pool, k_block) * qk_scale
        mask = (scores + theta) >= anchor_pool[:, None]
        valid_count = tl.sum(mask.to(tl.int32), axis=0, keep_dims=True)
        global_pos = tl.arange(0, BLOCK_SZ_N)[None, :] + pos
        selected = tl.where(mask, global_pos, N)
        selected = tl.sort(selected)
        count += valid_count
        tl.store(coords_ptr, selected)
        coords_ptr = tl.advance(coords_ptr, (0, valid_count))
        k_select_ptrs = tl.advance(k_select_ptrs, (0, BLOCK_SZ_N))

    # Store the counts of anchors per block
    tl.store(counts_ptr, count)


def get_mask_fused_anchor(
    Q, K, V,
    init_cnt, local_cnt,
    block_size, step,
    theta,
    B, H, N, h_dim, sm_scale,
    BLOCK_SZ_A=128, BLOCK_SZ_M=128, BLOCK_SZ_N=64
):
    # Ensure tensor shapes and contiguity
    assert Q.shape == K.shape == V.shape
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert init_cnt >= 0 and local_cnt >= 0
    assert BLOCK_SZ_M % block_size == 0
    assert (block_size * step) % BLOCK_SZ_M == 0
    assert (block_size * step) % BLOCK_SZ_N == 0
    assert N % (block_size * step) == 0

    # Allocate fp32 buffers for partial results and anchors
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device).contiguous()
    M_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device).contiguous()
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32).contiguous()
    true_coords = torch.zeros((B, H, N // block_size // step, N), dtype=torch.int32, device=Q.device).contiguous()
    true_counts = torch.zeros((B, H, N // block_size // step, 1), dtype=torch.int32, device=Q.device).contiguous()
    Logger.log(f"Device placements: Q:{Q.device}, K:{K.device}, V:{V.device}, Acc:{Acc_buffer.device}, L:{L_buffer.device}, M:{M_buffer.device}, coords:{true_coords.device}, counts:{true_counts.device}")

    # Launch Triton kernel across blocks and batch-heads
    grid = (triton.cdiv(N, block_size * step), B * H, 1)
    get_mask_fused_anchor_triton[grid](
        Q, K, V, L_buffer, M_buffer, Acc_buffer, true_coords, true_counts,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        L_buffer.stride(0), L_buffer.stride(1), M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        init_cnt, local_cnt, sm_scale, step, theta, B, H, N, h_dim,
        BLOCK_SZ=block_size, BLOCK_SZ_A=BLOCK_SZ_A, BLOCK_SZ_M=BLOCK_SZ_M, BLOCK_SZ_N=BLOCK_SZ_N,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8, num_stages=3
    )
    # Return buffers and computed anchor indices/counts
    return L_buffer, M_buffer, Acc_buffer, true_coords, true_counts
