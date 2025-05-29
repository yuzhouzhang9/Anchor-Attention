
import torch
import triton
import triton.language as tl
@triton.jit
def get_mask_fused_anchor_triton(Q, K, V, L_buffer, M_buffer, Acc_buffer, true_coords, true_counts,
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
                                dtype: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    if start_m * BLOCK_SZ * step >= N:
        return

    # Initialize offsets
    offs_m = start_m * (BLOCK_SZ * step) + tl.arange(0, BLOCK_SZ_M)
    offs_n = tl.arange(0, BLOCK_SZ_N)
    offs_d = tl.arange(0, D)
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    lm_offset = (off_hz // H) * stride_lb + (off_hz % H) * stride_lh
    mm_offset = (off_hz // H) * stride_mb + (off_hz % H) * stride_mh
    acc_offset = (off_hz // H) * stride_acc_z + (off_hz % H) * stride_acc_h
    local_offset = start_m * (BLOCK_SZ * step) - BLOCK_SZ_N
    local_offset = max(BLOCK_SZ_N, local_offset)
    l_ptrs = L_buffer + lm_offset + offs_m
    m_ptrs = M_buffer + mm_offset + offs_m
    acc_ptrs = Acc_buffer + acc_offset + offs_m[:, None] * stride_acc_m + offs_d[None, :] * stride_acc_k
    
    q_ptrs = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * (BLOCK_SZ * step), 0),
        block_shape=(BLOCK_SZ_M, D),
        order=(1, 0),
    )   
    k_init_ptrs = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SZ_N),
        order=(0, 1),
    )
    v_init_ptrs = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SZ_N, D),
        order=(1, 0),
    )
   
    
    qk_scale = sm_scale * 1.44269504
    m_i_tot = tl.zeros([BLOCK_SZ * step], dtype=tl.float32) - float("inf")
    q_avg_pool = tl.zeros([step, D], dtype=dtype)
    
    for i in range(0, BLOCK_SZ * step, BLOCK_SZ_M):
        q = tl.load(q_ptrs)
        q = (q * qk_scale).to(dtype)
        
        q_avg_block = tl.sum(q.reshape([BLOCK_SZ_M // BLOCK_SZ, BLOCK_SZ, D]), axis=1) / BLOCK_SZ
        new_q_avg_block = tl.broadcast_to(q_avg_block[None, :, :], (step // (BLOCK_SZ_M // BLOCK_SZ), BLOCK_SZ_M // BLOCK_SZ, D)).reshape([step, D])
        q_avg_block_arange = tl.arange(0, step) % (BLOCK_SZ_M // BLOCK_SZ) + (i // BLOCK_SZ)
        q_avg_pool = tl.where((tl.arange(0, step) == q_avg_block_arange)[:, None], new_q_avg_block.to(dtype), q_avg_pool)
        
        m_i = tl.zeros([BLOCK_SZ_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SZ_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_SZ_M, D], dtype=tl.float32)
        
        k_init = tl.load(k_init_ptrs)
        v_init = tl.load(v_init_ptrs)
        qk = tl.dot(q, k_init)
        cols = offs_n
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk += tl.where(causal_mask, qk, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v_init)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k_local_ptrs = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(D, N),
            strides=(stride_kd, stride_kn),
            offsets=(0, local_offset),
            block_shape=(D, BLOCK_SZ_N),
            order=(0, 1),
        )
        v_local_ptrs = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(N, D),
            strides=(stride_vn, stride_vd),
            offsets=(local_offset, 0),
            block_shape=(BLOCK_SZ_N, D),
            order=(1, 0),
        )
        for j in range(max(start_m * (BLOCK_SZ * step) - BLOCK_SZ_N, BLOCK_SZ_N), start_m * (BLOCK_SZ * step) + BLOCK_SZ_M + i, BLOCK_SZ_N):
            cols = tl.arange(0, BLOCK_SZ_N) + j
            k_local = tl.load(k_local_ptrs)
            v_local = tl.load(v_local_ptrs)
            qk = tl.dot(q, k_local)
            causal_mask = cols[None, :] <= offs_m[:, None]
            qk += tl.where(causal_mask, qk, float("-inf"))
            
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk - m_i_new[:, None])
            acc_scale = l_i * 0 + alpha
            acc *= acc_scale[:, None]
            acc += tl.dot(p.to(dtype), v_local)
            
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
            k_local_ptrs = tl.advance(k_local_ptrs, (0, BLOCK_SZ_N))
            v_local_ptrs = tl.advance(v_local_ptrs, (BLOCK_SZ_N, 0))
            
        new_m_i = tl.broadcast_to(m_i[None, :], (BLOCK_SZ * step // BLOCK_SZ_M, BLOCK_SZ_M)).reshape([BLOCK_SZ * step])
        m_i_arange = tl.arange(0, BLOCK_SZ * step) % BLOCK_SZ_M + i
        m_i_tot = tl.where(tl.arange(0, BLOCK_SZ * step) == m_i_arange, new_m_i, m_i_tot)
        
        q_ptrs = tl.advance(q_ptrs, (BLOCK_SZ_M, 0))
        tl.store(acc_ptrs, acc.to(Acc_buffer.type.element_ty))
        tl.store(l_ptrs, l_i.to(L_buffer.type.element_ty))
        tl.store(m_ptrs, m_i.to(M_buffer.type.element_ty))
        offs_m = offs_m + BLOCK_SZ_M
        l_ptrs = L_buffer + lm_offset + offs_m
        m_ptrs = M_buffer + mm_offset + offs_m
        acc_ptrs = Acc_buffer + acc_offset + offs_m[:, None] * stride_acc_m + offs_d[None, :] * stride_acc_k
    
    off_z = off_hz // H
    off_h = off_hz % H
    coords_offset = off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch
    counts_offset = off_z.to(tl.int64) * stride_tcz + off_h.to(tl.int64) * stride_tch
    
    coords_block_ptr = tl.make_block_ptr(
        base=true_coords + coords_offset,
        shape=(N // BLOCK_SZ // step, N),
        strides=(stride_cn, stride_cd),
        offsets=(start_m, 0),
        block_shape=(1, BLOCK_SZ_N),
        order=(1, 0),
    )
    counts_block_ptr = tl.make_block_ptr(
        base=true_counts + counts_offset,
        shape=(N // BLOCK_SZ // step, 1),
        strides=(stride_tcn, stride_tcd),
        offsets=(start_m, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )

    anchors_pool = tl.sum(tl.reshape(m_i_tot, [step, BLOCK_SZ]), axis=1) / BLOCK_SZ
    q_avg_pool = q_avg_pool.to(dtype)
    count = tl.zeros([1, 1], dtype=tl.int32)
    theta = theta * 1.44269504
    for i in range(BLOCK_SZ_N, start_m * (BLOCK_SZ * step) - BLOCK_SZ_N, BLOCK_SZ_N):
        k_init_ptrs = tl.advance(k_init_ptrs, (0, BLOCK_SZ_N))
        k = tl.load(k_init_ptrs)
        qk = tl.dot(q_avg_pool, k) * qk_scale
        cur_mask = ((qk + theta) >= anchors_pool[:, None]).to(tl.int32)
        f_mask = (tl.sum(cur_mask, axis=0, keep_dims=True).to(tl.int32) > 0)
        global_indices = tl.arange(0, BLOCK_SZ_N)[None, :] + i
        valid_mask = f_mask > 0
        valid_count = tl.sum(valid_mask.to(tl.int32))
        valid_positions = tl.where(valid_mask, global_indices, N)
        valid_positions = tl.sort(valid_positions)
        count = count + valid_count
        tl.store(coords_block_ptr, valid_positions)
        coords_block_ptr = tl.advance(coords_block_ptr, (0, valid_count))
    tl.store(counts_block_ptr, count)




def get_mask_fused_anchor(Q, K, V, init_cnt, local_cnt, block_size, step, theta, B, H, N, h_dim, sm_scale):
    assert Q.shape == K.shape == V.shape
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert init_cnt >= 0 and local_cnt >= 0
    assert block_size * step % 128 == 0
    assert N % (block_size * step) == 0
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device).contiguous()
    M_buffer = torch.full((B, H, N), -float("inf"), dtype=torch.float32, device=Q.device).contiguous()
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32).contiguous()
    true_coords = torch.zeros((B, H, N // block_size // step, N), dtype=torch.int32, device=Q.device).contiguous()
    true_counts = torch.zeros((B, H, N // block_size // step, 1), dtype=torch.int32, device=Q.device).contiguous()
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
        init_cnt, local_cnt, sm_scale, step, theta,
        B, H, N, h_dim,
        block_size,
        BLOCK_SZ_A=128,
        BLOCK_SZ_M=64,
        BLOCK_SZ_N=64,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.bfloat16,
        num_warps=8,
        num_stages=3,
    )
    return L_buffer, M_buffer, Acc_buffer, true_coords, true_counts
