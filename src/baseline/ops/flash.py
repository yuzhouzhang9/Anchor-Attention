import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

def gpu_info():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        return device_name, major
    return None, None


GPU_NAME, GPU_MAJOR = gpu_info()

def get_num_warps_stages(head_dim, block_size, gpu_name):
    """
    Returns recommended num_warps and num_stages for a Sparse Attention kernel in Triton.

    Args:
        head_dim (int): Size of the head dimension.
        block_size (int): Size of the block in the attention matrix.
        gpu_name (str): Name of the GPU.

    Returns:
        tuple: (num_warps, num_stages) recommended values.
    """
    gpu_name = gpu_name.lower()
    # Determine if head_dim and block_size exceed 64
    head_large = head_dim > 64
    block_large = block_size > 64

    if "h100" in gpu_name:
        # Hopper GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 4
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    elif "a100" in gpu_name:
        # Ampere GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 8
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    elif "4090" in gpu_name:
        if head_large and block_large:
            num_warps = 8
            num_stages = 2
        elif head_large or block_large:
            num_warps = 8
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    else:
        # use default setting, maybe not optimal
        if head_large and block_large:
            num_warps = 8
            num_stages = 2
        elif head_large or block_large:
            num_warps = 4
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    return num_warps, num_stages

@triton.jit
def prefill_kernel(
    q_ptr,  # Q: b x n x h x d
    k_ptr,  # K: b x n x h x d
    v_ptr,  # V: b x n x h x d
    o_ptr,
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    Q_LEN,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    # softmax_scale
    softmax_scale,
    # causal
    causal,
    # gqa
    gqa_interleave,
    # stride
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
):
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)
    # full attention or causal attention
    lo = 0
    if causal:
        hi = min(K_LEN, (pid_q + 1) * BLOCK_SIZE_Q)
    else:
        hi = K_LEN
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(1,), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        if causal:
            qk += tl.where(off_m[:, None] >= (i + off_n)[None, :], 0, float("-inf"))
        else:
            qk += tl.where((off_n < K_LEN - i)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    # final scale
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))

def triton_flash_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
):
    batch_size, q_len, num_q_heads, head_dim = q.shape
    batch_size, k_len, num_kv_heads, head_dim = k.shape
    assert v.shape == k.shape
    assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    # gqa
    assert num_q_heads % num_kv_heads == 0
    num_share_q_heads = num_q_heads // num_kv_heads
    # softmax_scale needs to be multiplied by math.log2(math.e)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # output tensor
    o = torch.zeros_like(q)

    grid = lambda META: (
        triton.cdiv(q_len, META["BLOCK_SIZE_Q"]),
        batch_size * num_q_heads,
    )
    # set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
    BLOCK_SIZE_Q = min(
        128, max(16, triton.next_power_of_2(q_len))
    )  # min block size of tl.dot: 16
    BLOCK_SIZE_K = 128
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, GPU_NAME)
    prefill_kernel[grid](
        q,
        k,
        v,
        o,
        batch_size,
        num_q_heads,
        num_kv_heads,
        num_share_q_heads,
        q_len,
        k_len,
        head_dim,
        softmax_scale,
        causal,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
@triton.jit
def decode_kernel(
    q_ptr,  # Q: b x 1 x h x d
    k_ptr,  # K: b x n x h x d
    v_ptr,  # V: b x n x h x d
    acco_ptr,  # acc_o: b x c x h x d
    lse_ptr,  # lse: b x c x h
    mi_ptr,  # mi: b x c x h
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    K_LEN,
    NUM_CHUNKS,
    HEAD_DIM: tl.constexpr,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave,
    # stride
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oc,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lc,
    stride_lh,
    stride_mb,
    stride_mc,
    stride_mh,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    CHUNK_SIZE_K: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE_K % BLOCK_SIZE_K == 0)
    # get batch id and head id
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    pid_c = tl.program_id(1)
    # init qkv pointer
    q_ptrs = (
        q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + tl.arange(0, HEAD_DIM) * stride_qd
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, pid_c * CHUNK_SIZE_K),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(pid_c * CHUNK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs)
    # init statistics
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((HEAD_DIM,), 0, dtype=tl.float32)
    # full attention
    lo = pid_c * CHUNK_SIZE_K
    hi = min(K_LEN, (pid_c + 1) * CHUNK_SIZE_K)
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(1,), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
        qk += tl.where((off_n < hi - i), 0, float("-inf"))
        qk += tl.sum(q[:, None] * k, axis=0) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.math.exp2(qk - m_ij)
        l_ij = tl.sum(p, axis=0)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale
        # load v and update acc_o
        v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.sum(p[:, None] * v, axis=0)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    # no final scale, do scale after all chunks are computed
    # acc_o = acc_o * tl.math.exp2(m_i - lse_i)
    # save lse and mi
    lse_ptr = (
        lse_ptr
        + pid_b * stride_lb
        + pid_h * stride_lh
        + (pid_c + tl.arange(0, 1)) * stride_lc
    )
    tl.store(lse_ptr, lse_i)
    mi_ptr = (
        mi_ptr
        + pid_b * stride_mb
        + pid_h * stride_mh
        + (pid_c + tl.arange(0, 1)) * stride_mc
    )
    tl.store(mi_ptr, m_i)
    # save chunk output
    off_d = tl.arange(0, HEAD_DIM)
    o_ptrs = (
        acco_ptr
        + pid_b * stride_ob
        + pid_c * stride_oc
        + pid_h * stride_oh
        + off_d * stride_od
    )
    tl.store(o_ptrs, acc_o)

@triton.jit
def rescale_kernel(
    acco_ptr,  # acc_o: b x c x h x d
    o_ptr,  # o: b x 1 x h x d
    lse_ptr,  # lse: b x c x h
    mi_ptr,  # mi: b x c x h
    # shape
    BATCH_SIZE,
    NUM_HEADS,
    NUM_CHUNKS,
    HEAD_DIM: tl.constexpr,
    # stride
    stride_ab,
    stride_ac,
    stride_ah,
    stride_ad,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lc,
    stride_lh,
    stride_mb,
    stride_mc,
    stride_mh,
    # META parameters
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # get batch id and head id
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    # ptrs
    off_chunks = tl.arange(0, BLOCK_SIZE_C)
    mi_ptrs = mi_ptr + pid_b * stride_mb + pid_h * stride_mh + off_chunks * stride_mc
    lse_ptrs = lse_ptr + pid_b * stride_lb + pid_h * stride_lh + off_chunks * stride_lc
    acco_ptrs = tl.make_block_ptr(
        base=acco_ptr + pid_b * stride_ab + pid_h * stride_ah,
        shape=(NUM_CHUNKS, HEAD_DIM),
        strides=(stride_ac, stride_ad),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_C, BLOCK_SIZE_D),
        order=(1, 0),
    )
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(1, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(0, 0),
        block_shape=(1, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # load mi and lse
    mi = tl.load(mi_ptrs, mask=off_chunks < NUM_CHUNKS, other=float("-inf"))
    lse = tl.load(lse_ptrs, mask=off_chunks < NUM_CHUNKS, other=float("-inf"))
    # get scale factor
    m = tl.max(mi, axis=0)
    scale = tl.math.exp2(mi - m) / tl.sum(tl.math.exp2(lse - m), axis=0)
    # reduce
    o = tl.full((HEAD_DIM,), 0, dtype=tl.float32)
    for i in range(0, HEAD_DIM, BLOCK_SIZE_D):
        i = tl.multiple_of(i, BLOCK_SIZE_D)
        # rescale and reduce
        acco = tl.load(acco_ptrs, boundary_check=(0, 1), padding_option="zero")
        acco = tl.sum(acco * scale[:, None], axis=0)[None, :]
        # save
        tl.store(o_ptrs, acco.to(tl.bfloat16), boundary_check=(0, 1))
        # update ptrs
        acco_ptrs = tl.advance(acco_ptrs, (0, BLOCK_SIZE_D))
        o_ptrs = tl.advance(o_ptrs, (0, BLOCK_SIZE_D))



def triton_flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
):
    batch_size, q_len, num_q_heads, head_dim = q.shape
    batch_size, k_len, num_kv_heads, head_dim = k.shape
    assert q_len == 1
    assert v.shape == k.shape
    assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    # softmax_scale needs to be multiplied by math.log2(math.e)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # gqa
    assert num_q_heads % num_kv_heads == 0
    num_share_q_heads = num_q_heads // num_kv_heads
    # grid
    grid = lambda META: (
        batch_size * num_q_heads,  # batch & head
        triton.cdiv(k_len, META["CHUNK_SIZE_K"]),  # k chunks
    )
    # set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
    BLOCK_SIZE_K = 128
    CHUNK_SIZE_K = 4096
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, GPU_NAME)
    # chunk output and chunk lse and chunk
    num_chunks = triton.cdiv(k_len, CHUNK_SIZE_K)
    lse = torch.empty(
        batch_size, num_chunks, num_q_heads, dtype=torch.float32, device=q.device
    )
    mi = torch.empty(
        batch_size, num_chunks, num_q_heads, dtype=torch.float32, device=q.device
    )
    acc_o = torch.empty(
        batch_size,
        num_chunks,
        num_q_heads,
        head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    # launch kernel
    decode_kernel[grid](
        q,
        k,
        v,
        acc_o,
        lse,
        mi,
        batch_size,
        num_q_heads,
        num_kv_heads,
        num_share_q_heads,
        k_len,
        num_chunks,
        head_dim,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        acc_o.stride(0),
        acc_o.stride(1),
        acc_o.stride(2),
        acc_o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        mi.stride(0),
        mi.stride(1),
        mi.stride(2),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        CHUNK_SIZE_K=CHUNK_SIZE_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # rescale
    o = torch.empty(
        batch_size,
        1,
        num_q_heads,
        head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    # grid
    grid = lambda META: (batch_size * num_q_heads,)  # batch & head
    # set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
    BLOCK_SIZE_C = triton.next_power_of_2(num_chunks)
    BLOCK_SIZE_D = min(head_dim, 128 * 128 // BLOCK_SIZE_C)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, GPU_NAME)
    # launch kernel
    rescale_kernel[grid](
        acc_o,
        o,
        lse,
        mi,
        batch_size,
        num_q_heads,
        num_chunks,
        head_dim,
        acc_o.stride(0),
        acc_o.stride(1),
        acc_o.stride(2),
        acc_o.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        mi.stride(0),
        mi.stride(1),
        mi.stride(2),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o




def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
):
    batch_size, q_len, num_heads, head_dim = q.shape
    batch_size, k_len, num_heads, head_dim = k.shape
    assert v.shape == k.shape
    assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
    assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
    if q_len > 1:
        return triton_flash_prefill(q, k, v, causal, softmax_scale, gqa_interleave)
    else:
        return triton_flash_decode(q, k, v, softmax_scale, gqa_interleave)


if __name__ == "__main__":
    device = torch.device('cuda:1')
    torch.manual_seed(42)
    
    # Test config: batch=2, q_len=128, k_len=128, heads=8, head_dim=64
    batch_size, q_len, k_len, num_heads, head_dim = 2, 128, 128, 8, 64
    
    # Create test inputs
    q = torch.randn(batch_size, q_len, num_heads, head_dim, device=device).to(torch.bfloat16)
    k = torch.randn(batch_size, k_len, num_heads, head_dim, device=device).to(torch.bfloat16)
    v = torch.randn(batch_size, k_len, num_heads, head_dim, device=device).to(torch.bfloat16)

    triton_out = triton_flash_attention(q, k, v, causal=True)
    