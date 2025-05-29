import torch
import triton
import triton.language as tl
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import time
from my_utils import Logger

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
    # Determine the block index for queries and the combined batch-head index
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_SZ_N)

    # Compute offsets along batch and head dimensions
    off_z = off_hz // H
    off_h = off_hz % H
    qo_offset = off_z * stride_qz + off_h * stride_qh
    kv_offset = off_z * stride_kz + off_h * stride_kh
    lm_offset = off_z * stride_lb + off_h * stride_lh
    mm_offset = off_z * stride_mb + off_h * stride_mh
    acc_offset = off_z * stride_acc_z + off_h * stride_acc_h

    # Identify global block boundaries for initialization and local scopes
    big_block_startm = (start_m * BLOCK_SZ_M) // (BLOCK_SZ * step)
    init_offset = min(init_cnt * BLOCK_SZ * step, (start_m + 1) * BLOCK_SZ_M)
    local_offset = max((big_block_startm - local_cnt + 1) * BLOCK_SZ * step, 0)

    # Create block pointers for Q, L, M, Acc buffers
    q_ptrs = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N, D),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_SZ_M, 0),
        block_shape=(BLOCK_SZ_M, D),
        order=(1, 0)
    )
    l_ptrs = tl.make_block_ptr(
        base=L_buffer + lm_offset,
        shape=(N,),
        strides=(1,),
        offsets=(start_m * BLOCK_SZ_M,),
        block_shape=(BLOCK_SZ_M,),
        order=(0,)
    )
    m_ptrs = tl.make_block_ptr(
        base=M_buffer + mm_offset,
        shape=(N,),
        strides=(1,),
        offsets=(start_m * BLOCK_SZ_M,),
        block_shape=(BLOCK_SZ_M,),
        order=(0,)
    )
    acc_ptrs = tl.make_block_ptr(
        base=Acc_buffer + acc_offset,
        shape=(N, D),
        strides=(stride_acc_m, stride_acc_k),
        offsets=(start_m * BLOCK_SZ_M, 0),
        block_shape=(BLOCK_SZ_M, D),
        order=(1, 0)
    )

    # Block pointers for initial and local key/value windows
    k_init_ptrs = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(D, BLOCK_SZ_N),
        order=(0, 1)
    )
    v_init_ptrs = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SZ_N, D),
        order=(1, 0)
    )
    k_local_ptrs = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, local_offset),
        block_shape=(D, BLOCK_SZ_N),
        order=(0, 1)
    )
    v_local_ptrs = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(local_offset, 0),
        block_shape=(BLOCK_SZ_N, D),
        order=(1, 0)
    )

    # Pre-scale queries for log-domain softmax
    q = tl.load(q_ptrs)
    q = (q * sm_scale * 1.44269504).to(dtype)
    offs_m = start_m * BLOCK_SZ_M + tl.arange(0, BLOCK_SZ_M)

    # Initialize running max (m_i), log-sum-exp accumulator (l_i), and output accumulator
    m_i = tl.zeros([BLOCK_SZ_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SZ_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SZ_M, D], dtype=tl.float32)

    # Process initial global window
    for j in range(0, init_offset, BLOCK_SZ_N):
        k_init = tl.load(k_init_ptrs).to(dtype)
        qk = tl.dot(q, k_init)
        causal_mask = (offs_n[None, :] + j) <= offs_m[:, None]
        qk = qk + tl.where(causal_mask, 0, -1e6)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v_init = tl.load(v_init_ptrs)
        acc += tl.dot(p.to(dtype), v_init)
        m_i = m_new
        k_init_ptrs = tl.advance(k_init_ptrs, (0, BLOCK_SZ_N))
        v_init_ptrs = tl.advance(v_init_ptrs, (BLOCK_SZ_N, 0))

    # Process local window around each query block
    for j in range(max(local_offset, init_offset), (big_block_startm + 1) * BLOCK_SZ * step, BLOCK_SZ_N):
        k_local = tl.load(k_local_ptrs)
        qk = tl.dot(q, k_local)
        causal_mask = (offs_n[None, :] + j) <= offs_m[:, None]
        qk = qk + tl.where(causal_mask, 0, -1e6)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v_local = tl.load(v_local_ptrs)
        acc += tl.dot(p.to(dtype), v_local)
        m_i = m_new
        k_local_ptrs = tl.advance(k_local_ptrs, (0, BLOCK_SZ_N))
        v_local_ptrs = tl.advance(v_local_ptrs, (BLOCK_SZ_N, 0))

    # Store updated Acc, L, M buffers
    tl.store(acc_ptrs, acc.to(Acc_buffer.type.element_ty))
    tl.store(l_ptrs, l_i.to(L_buffer.type.element_ty))
    tl.store(m_ptrs, m_i.to(M_buffer.type.element_ty))


def anchor_attn_step_1(
    Q, K, V,
    init_cnt,
    local_cnt,
    block_size,
    step,
    B, H, N, D, sm_scale,
    BLOCK_SZ_M=128, BLOCK_SZ_N=128
):
    # Allocate buffers for L, M, and Accumulator (in fp32)
    L_buffer = torch.zeros((B, H, N), dtype=torch.float32, device=Q.device).contiguous()
    M_buffer = torch.full((B, H, N), -float("inf"), dtype=torch.float32, device=Q.device).contiguous()
    Acc_buffer = torch.zeros_like(Q, dtype=torch.float32).contiguous()
    grid = (triton.cdiv(N, BLOCK_SZ_M), B * H, 1)
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
        BLOCK_SZ_M=BLOCK_SZ_M,
        BLOCK_SZ_N=BLOCK_SZ_N,
        dtype=tl.bfloat16 if Q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=3,
    )
    return L_buffer, M_buffer, Acc_buffer


def anchor_attn_step_2(
    Q, K, M_buffer,
    block_size,
    step,
    theta, B, H, N, D, sm_scale
):
    from .get_mask_test import get_q_mask
    return get_q_mask(
        Q, K, M_buffer.unsqueeze(-1),
        block_size, step, theta,
        B, H, N, D, sm_scale
    )

if __name__ == "__main__":
    # Quick precision and speed checks
    B, H, N, D = 1, 32, 128*1024, 128
    BLOCK_SZ, step, theta = 128, 16, 0
    sm_scale = D ** -0.5
    init_cnt, local_cnt = 1, 1
    Q = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)
    V = torch.randn((B, H, N, D), device="cuda", dtype=torch.bfloat16)

    L1, M1, Acc1 = anchor_attn_step_1(Q, K, V, init_cnt, local_cnt, BLOCK_SZ, step, B, H, N, D, sm_scale)
    from .anchor_step_1_2 import anchor_attn_step_1 as anchor_attn_step_1_v1
    L2, M2, Acc2, _, _ = anchor_attn_step_1_v1(Q, K, V, init_cnt, local_cnt, BLOCK_SZ, step, theta, B, H, N, D, sm_scale)
    print("[info] Precision differences:", torch.max(torch.abs(L1 - L2)), torch.max(torch.abs(M1 - M2)), torch.max(torch.abs(Acc1 - Acc2)))

    print("[info] Speed comparison:")
    import time
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        anchor_attn_step_1(Q, K, V, init_cnt, local_cnt, BLOCK_SZ, step, B, H, N, D, sm_scale)
        mask, mask_cnt = anchor_attn_step_2(Q, K, M1, BLOCK_SZ, step, theta, B, H, N, D, sm_scale)
        torch.cuda.synchronize()
        print(f"V2 time: {(time.time() - t0)*1000:.2f}ms")
        torch.cuda.synchronize()
        t0 = time.time()
        anchor_attn_step_1_v1(Q, K, V, init_cnt, local_cnt, BLOCK_SZ, step, theta, B, H, N, D, sm_scale)
        torch.cuda.synchronize()
        print(f"V1 time: {(time.time() - t0)*1000:.2f}ms")
