# return L, M, ACC, mask
# Q[B, H, seq, h_dim], K[B, H, seq, h_dim], V[B, H, seq, h_dim], init_cnt[B, H], local_cnt[B, H]
import torch
import triton
import triton.language as tl
 
 
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
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_INDICES: tl.constexpr,
    dtype: tl.constexpr,
):
    # Program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Get sequence length
    seqlen = tl.load(seqlens + off_hz // H)
    query_start = start_m * STEP
    if query_start >= seqlen:
        return

    # Initialize offsets
    offs_m = query_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute memory offsets
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    lm_offset = (off_hz // H) * stride_lb + (off_hz % H) * stride_lh
    ab_offset = (off_hz // H) * stride_abz + (off_hz % H) * stride_abh
    tc_offset = (off_hz // H) * stride_tcz + (off_hz % H) * stride_tch + start_m * stride_tcm
    tct_offset = (off_hz // H) * stride_tctz + (off_hz % H) * stride_tcth + start_m * stride_tctm

    # Pointers
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    
    tc_ptrs = true_coords + tc_offset
    tct_ptrs = true_counts + tct_offset

    # Load existing l_i, m_i, and acc
    

    # Scale factor
    qk_scale = sm_scale * 1.44269504

    # Load number of sparse key blocks
    sparse_count = tl.load(tct_ptrs)

    for M_idex in range(0, STEP, BLOCK_M):
        offs_m = query_start + tl.arange(0, BLOCK_M) + M_idex
        q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        l_ptrs = L_buffer + lm_offset + offs_m
        m_ptrs = M_buffer + lm_offset + offs_m
        a_ptrs = Acc_buffer + ab_offset + offs_m[:, None] * stride_abm + offs_d[None, :] * stride_abd
        
        m_i = tl.load(m_ptrs, mask=offs_m < seqlen, other=-float("inf"))
        l_i = tl.load(l_ptrs, mask=offs_m < seqlen, other=0.0)
        acc = tl.load(a_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
        q = (q * qk_scale).to(dtype)

        # 
        for idx_start in range(0, sparse_count, NUM_INDICES):
            offs_idx = idx_start + tl.arange(0, NUM_INDICES)
            idx_mask = offs_idx < sparse_count
            real_block_idx = tl.load(tc_ptrs + offs_idx, mask=idx_mask, other=0)
            offs_n = real_block_idx
            key_mask = offs_n < seqlen
            combined_mask = idx_mask & key_mask
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
        
        tl.store(o_ptrs, acc.to(dtype), mask=offs_m[:, None] < seqlen)




# step * BLOCK_N
@triton.jit
def _get_kmask_triton(Q, K, M_buffer, q_mask, 
                    stride_qz, stride_qh, stride_qm, stride_qd,
                    stride_kz, stride_kh, stride_kn, stride_kd,
                    stride_mz, stride_mh, stride_mn, stride_md,
                    stride_qmz, stride_qmh, stride_qmm, stride_qmd,
                    sm_scale:tl.constexpr, step: tl.constexpr, diff: tl.constexpr, ratio: tl.constexpr,
                    B, H, N: tl.constexpr, D: tl.constexpr,
                    BLOCK_M:tl.constexpr, BLOCK_N: tl.constexpr, dtype: tl.constexpr):
    
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    if start_m * BLOCK_M >= N:
        return 
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    mb_offset = (off_hz // H) * stride_mz + (off_hz % H) * stride_mh
    qm_offset = (off_hz // H) * stride_qmz + (off_hz % H) * stride_qmh
    q_ptrs = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )    
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_N * step),
        order=(0, 1),
    )
    M_buffer_ptr = tl.make_block_ptr(
        base=M_buffer + mb_offset,
        shape = (N, 1),
        strides=(stride_mn, stride_md),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    qmask_ptrs = tl.make_block_ptr(
        base=q_mask + qm_offset,
        shape=(N, 1),
        strides=(stride_qmm, stride_qmd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    m_i = tl.load(M_buffer_ptr)
    q = (q * qk_scale).to(dtype)
    offs_n = tl.arange(0, BLOCK_N * step)
    mask_count = tl.zeros([BLOCK_M], dtype=tl.int32)
    for i in range(0, start_m * BLOCK_M, BLOCK_N * step):
        k = tl.load(K_block_ptr) # [D, BLOCK_N * step]
        k = tl.reshape(k, (D, BLOCK_N, step))  
        rk = tl.sum(k, axis=1, keep_dims=False) / BLOCK_N
        qk = tl.dot(q, rk)  
        cur_mask = ((qk + diff) >= m_i).to(tl.int32)   # [BM, step]
        mask_count += tl.sum(cur_mask, axis=1, keep_dims=False) # [BM]
    
    f_mask = tl.where(mask_count >= start_m * BLOCK_M * ratio, 1, 0) # [BM]
    tl.store(qmask_ptrs, f_mask.to(q_mask.type.element_ty))
   #MdN means BLOCK_M = MdN * BLOCK_N  


@triton.jit
def _get_mask_fused_anchor_triton(Q, K, V, L_buffer, M_buffer, Acc_buffer, true_coords, true_counts, 
                                stride_qz, stride_qh, stride_qm, stride_qd,
                                stride_kz, stride_kh, stride_kn, stride_kd,
                                stride_vz, stride_vh, stride_vn, stride_vd,
                                stride_lb, stride_lh, stride_mb, stride_mh,
                                stride_acc_z, stride_acc_h, stride_acc_m, stride_acc_k,  # Strides for Acc_buffer
                                stride_cz, stride_ch, stride_cn, stride_cd,
                                stride_tcz, stride_tch, stride_tcn, stride_tcd, 
                                init_cnt:tl.constexpr, local_cnt:tl.constexpr, sm_scale:tl.constexpr, step: tl.constexpr, diff: tl.constexpr, 
                                B, H, N: tl.constexpr, D: tl.constexpr,
                                BLOCK_M:tl.constexpr, BLOCK_N: tl.constexpr, NdM: tl.constexpr, dtype: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    if start_m * BLOCK_M * step >= N:
        return 
    # Initialize offsets
    offs_m = start_m * (BLOCK_M * step) + tl.arange(0, BLOCK_M * step)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    lm_offset = (off_hz // H) * stride_lb + (off_hz % H) * stride_lh
    mm_offset = (off_hz // H) * stride_mb + (off_hz % H) * stride_mh 
    acc_offset = (off_hz // H) * stride_acc_z + (off_hz % H) * stride_acc_h  # Offset for Acc_buffer
    local_start = (start_m + 1) * step // NdM - local_cnt # local window 开始的 kblock 左闭
    local_offset = (start_m + 1) * (BLOCK_M * step) - (local_cnt * BLOCK_N)
    local_offset = max(0, local_offset)
    l_ptrs = L_buffer + lm_offset + offs_m
    m_ptrs = M_buffer + mm_offset + offs_m
    acc_ptrs = Acc_buffer + acc_offset + offs_m[:, None] * stride_acc_m + offs_d[None, :] * stride_acc_k  # Pointer for Acc_buffer
    m_i = tl.zeros([BLOCK_M * step], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M * step], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M * step, D], dtype=tl.float32)  
    
    q_ptrs = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * (BLOCK_M * step), 0),
        block_shape=(BLOCK_M * step, D),
        order=(1, 0),
    )    
    k_init_ptrs = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_N),
        order=(0, 1),
    )
    v_init_ptrs = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, D),
        order=(1, 0),
    )
    k_local_ptrs = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, local_offset),
        block_shape=(D, BLOCK_N),
        order=(0, 1),
    )
    v_local_ptrs = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(local_offset, 0),
        block_shape=(BLOCK_N, D),
        order=(1, 0),
    )
 
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)
     
     
    # for the BLOCK_N count need to be caculated: 
    for i in range(0, min(init_cnt, (start_m + 1) * step)  // NdM):
        # Load K and V
        cols = offs_n + i * BLOCK_N
        k_init = tl.load(k_init_ptrs)
        v_init = tl.load(v_init_ptrs)  
        qk = tl.dot(q, k_init) # (BLOCK_M * step, BLOCK_N)
        # cols[BN] offs_m[BM * step]
        causal_mask = cols[None, :] <= offs_m[:, None] 
        qk += tl.where(causal_mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v_init)
        
        # Update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k_init_ptrs = tl.advance(k_init_ptrs,(0, BLOCK_N))
        v_init_ptrs = tl.advance(v_init_ptrs,(BLOCK_N, 0))
    
    for i in range(max(init_cnt, local_start), (start_m + 1) * step // NdM):
        # Load K and V
        cols = offs_n + i * BLOCK_N
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
        
        # Update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k_local_ptrs = tl.advance(k_local_ptrs,(0, BLOCK_N))
        v_local_ptrs = tl.advance(v_local_ptrs,(BLOCK_N, 0))
    
    tl.store(acc_ptrs, acc.to(Acc_buffer.type.element_ty))
    tl.store(l_ptrs, l_i.to(L_buffer.type.element_ty))
    tl.store(m_ptrs, m_i.to(M_buffer.type.element_ty))
    
    # Load K and V
    off_z = off_hz // H
    off_h = off_hz % H
    r_q = tl.reshape(q, (step, BLOCK_M, D))  # [step, BS, D] # [16, 16, 128]
    r_q = tl.sum(r_q, axis=1, keep_dims=False) / BLOCK_M  # [step, D] # [16, 128]
    r_q = r_q.to(tl.float16)
    coords_offset = off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch
    counts_offset = off_z.to(tl.int64) * stride_tcz + off_h.to(tl.int64) * stride_tch
    anchors = m_i
    anchors = tl.reshape(anchors, (step, BLOCK_M, 1))  # [step, BS, 1]
    anchors = tl.sum(anchors, axis=1, keep_dims=False) / BLOCK_M # [step, 1] #[16, 1]
    anchors = anchors.to(tl.float16)
    
    coords_block_ptr = tl.make_block_ptr(
        base=true_coords + coords_offset,
        shape=(N // BLOCK_M // step, N), #[1,256]
        strides=(stride_cn, stride_cd),
        offsets=(start_m, 0),
        block_shape=(1, BLOCK_N), #[1, 32]
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, init_cnt),
        block_shape=(D, BLOCK_N),
        order=(0, 1),
    )
    counts_block_ptr = tl.make_block_ptr(
        base=true_counts + counts_offset,
        shape=(N // BLOCK_M // step, 1),
        strides=(stride_tcn, stride_tcd),
        offsets=(start_m, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )   
    count = tl.zeros((1,1), dtype=tl.int32)  # 记录有效索引数量
    cur_mask = tl.zeros((step, BLOCK_N), dtype=tl.int32)
    f_mask = tl.zeros((1, BLOCK_N), dtype=tl.int1) 
    for i in range(init_cnt, local_start):
        k = tl.load(K_block_ptr)
        rqk = tl.dot(r_q, k) * qk_scale  # [step, BS]
        rqk = rqk.to(tl.float16) 
        cur_mask = ((rqk + diff) >= anchors).to(tl.int32)   # [step, BS]
        f_mask = (tl.sum(cur_mask, axis=0, keep_dims=True).to(tl.int32) > 0)   # [1, BS]
        global_indices = tl.arange(0, BLOCK_N)[None, :] + (i * BLOCK_N)  # [1, BS]
        valid_mask = f_mask > 0  # [1, BS]
        valid_count = tl.sum(valid_mask.to(tl.int32))  # Scalar
        valid_positions = tl.where(valid_mask, global_indices, N)  # [1, BS] 
        valid_positions = tl.sort(valid_positions)  # Sort indices, not the mask
        count = count + valid_count 
        tl.store(coords_block_ptr, valid_positions.to(true_coords.type.element_ty))
        coords_block_ptr = tl.advance(coords_block_ptr, (0, valid_count))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    tl.store(counts_block_ptr, count)
    

is_print = True
def anchor_attn(Q, K, V, init_cnt, local_cnt, block_m, block_n, step, diff, B, H, N, h_dim, sm_scale):
    assert Q.shape == K.shape == V.shape
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert init_cnt >= 0 and local_cnt >= 0
    assert init_cnt * block_n <= N and local_cnt * block_n <= N
    assert step >= 16
    assert local_cnt * block_n >= step * block_m
    assert block_n % block_m == 0
    NdM = block_n // block_m
    assert step % NdM == 0    
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device).contiguous()
    M_buffer = torch.full((B, H, N), -float("inf"), dtype=torch.float32, device=Q.device).contiguous() # as anchors
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32).contiguous()  
    true_coords = torch.empty((B, H, N // block_m // step, N), dtype=torch.int32, device=Q.device).contiguous()
    true_counts = torch.empty((B, H, N // block_m // step, 1), dtype=torch.int32, device=Q.device).contiguous()
    
        
    grid = (triton.cdiv(N, block_m * step), B * H, 1)
    global is_print
    if is_print:
        print("[info] Q.shape: ", Q.shape)
        print("[info] grid:", grid)
        print("[info] NdM:", NdM)
        print("[info] block_m:", block_m)
        print("[info] block_n:", block_n)
        print("[info] step:", step)
        print("[info] diff:", diff)
        print("[info] sm_scale:", sm_scale)
        print("[info] init_cnt:", init_cnt)
        print("[info] local_cnt:", local_cnt)
        is_print = False
        
    _get_mask_fused_anchor_triton[grid](
        Q,K,V,L_buffer,M_buffer,Acc_buffer,true_coords,true_counts,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        L_buffer.stride(0), L_buffer.stride(1), M_buffer.stride(0), M_buffer.stride(1),
        Acc_buffer.stride(0), Acc_buffer.stride(1), Acc_buffer.stride(2), Acc_buffer.stride(3),
        true_coords.stride(0), true_coords.stride(1), true_coords.stride(2), true_coords.stride(3),
        true_counts.stride(0), true_counts.stride(1), true_counts.stride(2), true_counts.stride(3),
        init_cnt, local_cnt, sm_scale, step, diff,
        B, H, N, h_dim,
        block_m, block_n, NdM, dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3,     
    )
    q_mask = torch.zeros((B, H, N // (block_m * step), 1), dtype=torch.int32, device=Q.device).contiguous()
    grid = (triton.cdiv(N, (block_m * step)), B * H, 1)
    _get_kmask_triton[grid](
        Q, K, M_buffer, q_mask,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        M_buffer.stride(0), M_buffer.stride(1), M_buffer.stride(2), M_buffer.stride(3),
        stride_qmz=Q.stride(0), stride_qmh=Q.stride(1), stride_qmm=Q.stride(2), stride_qmd=Q.stride(3),
        sm_scale=sm_scale, step=step, diff=diff, ratio=0.8,
        B=B, H=H, N=N, D=h_dim, 
    )
    print(true_coords)
    # print(true_coords.count_nonzero(dim=-1, dtype=torch.int32))
    print(true_counts)
    seqlens = torch.full((B,), N , dtype=torch.int32, device=Q.device)
    NUM_ROWS_STEP2 = (N // block_m + step - 1) // step
    MAX_BLOCKS_PRE_ROW_STEP2 = true_coords.shape[-1]
    grid_step2 = (NUM_ROWS_STEP2, B * H, 1)
    output = torch.zeros_like(Q, device=Q.device)
    
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
        STEP = step * block_m, 
        BLOCK_M = 128, # config
        BLOCK_N = 32,
        BLOCK_DMODEL=D,
        NUM_INDICES=64,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3
    )
    sparsity_ratio = true_counts.sum() / (B * H * N / (block_m * step)* N// 2)
    print(f"anchor sparsity_ratio:{sparsity_ratio}")
    return output



if __name__ == "__main__":
    B, H, N, D = 1, 1,  1024, 128
    BLOCK_M = 16
    step = 16
    BLOCK_N = 64 
    diff = 3.58   
    sm_scale = D ** -0.5
    init_cnt = 1
    local_cnt = 4
    loop_time = 1
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    exe_time = 0x3f3f3f3f
    for i in range(loop_time): 
        Q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        K = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        V = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        start_event.record()
        O = anchor_attn(Q, K, V, init_cnt, local_cnt, BLOCK_M, BLOCK_N, step, diff, B, H, N, D, sm_scale)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        exe_time = min(exe_time, elapsed_time_ms)
    # print(O)
    print(f"Kernel execution time: {exe_time:.3f} ms")
 