import torch
import triton
import triton.language as tl
import time
from my_utils import Logger
# ------------------------------
# Triton kernel definition
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
    # Identify the row block and head-offset
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Load sequence length for this head
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # Compute row, column, and feature offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute base offsets for Q and K/V pointers
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    # Build pointer arrays for Q, K, V, and output tensors
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # Compute pointer to the sparse block index array
    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

    # Initialize accumulation buffers for stable softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # Convert to base-2 exponent scaling

    # Load and scale query vectors
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # Create a mask for valid query rows
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)

    # Iterate over selected sparse blocks
    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)

        # Compute raw attention scores with causal masking
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        if BLOCK_N >= 16:
            qk += tl.dot(q, k)
        else:
            qk += tl.sum(q[:, None, :] * k.T[None, :, :], axis=2)

        # Update running maximum for numerical stability
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # Rescale accumulated values
        acc *= alpha[:, None]
        if BLOCK_N >= 16:
            acc += tl.dot(p.to(dtype), v)
        else:
            acc += tl.sum(p[:, :, None] * v[None, :, :], axis=1)

        # Update normalization factors
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Final output normalization
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)

# ------------------------------
# Core Triton block-sparse attention function
# ------------------------------
def _triton_block_sparse_attention(
    q, k, v, seqlens, block_index, sm_scale,
    block_size_M=128, block_size_N=128,
) -> torch.Tensor:
    """
    Executes the Triton kernel to compute block-sparse attention.

    Parameters:
        q, k, v: Query, Key, Value tensors of shape [B, H, L, D]
        seqlens: Sequence lengths per batch-head
        block_index: Indices of selected key blocks per query block
        sm_scale: Scaling factor for softmax
        block_size_M, block_size_N: Block sizes for query and key/value

    Returns:
        Output tensor of same shape as q
    """
    # Ensure consistent feature dimensions
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk == Lv
    o = torch.zeros_like(q)

    # Configure Triton launch grid: one program per query block and head
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_block_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale, block_index, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_index.shape[-2], block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk, dtype=dtype,
        num_warps=8, num_stages=3,
    )
    return o

# ------------------------------
# Block index builders for different selection strategies
# ------------------------------

def _build_block_index(
    query: torch.Tensor,
    key: torch.Tensor,
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
):
    """
    Selects the top-k key blocks per query block based on mean-pooled dot-products.
    Returns block indices and sparsity ratio.
    """
    batch_size, num_heads, context_size, head_dim = query.shape
    num_M_blocks = (context_size + block_size_M - 1) // block_size_M
    num_N_blocks = (context_size + block_size_N - 1) // block_size_N

    # Mean-pool queries and keys into blocks
    query_pool = query.view(batch_size, num_heads, num_M_blocks, block_size_M, head_dim).mean(dim=-2)
    key_pool = key.view(batch_size, num_heads, num_N_blocks, block_size_N, head_dim).mean(dim=-2)

    # Compute score matrix and apply causal masking
    arange_M = torch.arange(num_M_blocks, device=query.device) * block_size_M
    arange_N = torch.arange(num_N_blocks, device=query.device) * block_size_N
    p_pool = torch.einsum('bhmk,bhnk->bhmn', query_pool, key_pool)
    p_pool = p_pool.masked_fill(arange_M[None,None,:,None] < arange_N[None,None,None,:], float('-inf'))

    # Select top_k blocks per row
    top_k = min(top_k, num_N_blocks)
    indices = torch.topk(p_pool, top_k, dim=-1).indices
    block_index = indices.sort(dim=-1).values

    # Compute sparsity ratio
    total_possible = batch_size * num_heads * num_M_blocks * (num_N_blocks + 1) // 2
    total_used = batch_size * num_heads * ((num_M_blocks - top_k) * top_k + top_k * (top_k + 1) // 2)
    sparsity_ratio = total_used / total_possible if total_possible > 0 else 0.0

    from my_utils import Logger
    Logger.log(f"total_used:{total_used}, possible:{total_possible}")
    return block_index.int(), sparsity_ratio


def _build_block_index_top_cdf(
    query: torch.Tensor,
    key: torch.Tensor,
    cdf_threshold: float = 0.95,
    block_size_M: int = 128,
    block_size_N: int = 128,
    sm_scale: float = None,
):
    """
    Selects key blocks whose cumulative softmax probability exceeds cdf_threshold.
    Returns block indices and sparsity ratio.
    """
    batch_size, num_heads, context_size, head_dim = query.shape
    num_M_blocks = (context_size + block_size_M - 1) // block_size_M
    num_N_blocks = (context_size + block_size_N - 1) // block_size_N

    # Mean-pool blocks and scale queries
    query_pool = query.view(batch_size, num_heads, num_M_blocks, block_size_M, head_dim).mean(dim=-2) * sm_scale
    key_pool = key.view(batch_size, num_heads, num_N_blocks, block_size_N, head_dim).mean(dim=-2)

    arange_M = torch.arange(num_M_blocks, device=query.device) * block_size_M
    arange_N = torch.arange(num_N_blocks, device=query.device) * block_size_N
    p_pool = torch.einsum('bhmk,bhnk->bhmn', query_pool, key_pool)
    p_pool = p_pool.masked_fill(arange_M[None,None,:,None] < arange_N[None,None,None,:], float('-inf'))

    # Compute CDF and find cutoff
    p_softmax = torch.softmax(p_pool, dim=-1)
    cdf = p_softmax.cumsum(dim=-1)
    cutoff_idx = (cdf > cdf_threshold).float().argmax(dim=-1) + 1

    # Build boolean mask of selected blocks
    n_idx = torch.arange(num_N_blocks, device=query.device)
    mask = (n_idx[None,None,None,:] < cutoff_idx[...,None]) & \
           (arange_M[None,None,:,None] >= arange_N[None,None,None,:])

    # Pack indices into a dense tensor
    max_blocks = mask.sum(dim=-1).max().item()
    block_index = torch.zeros(batch_size, num_heads, num_M_blocks, max_blocks, dtype=torch.int64, device=query.device)
    total_used = 0
    for b in range(batch_size):
        for h in range(num_heads):
            for m in range(num_M_blocks):
                sel = n_idx[mask[b,h,m]]
                count = sel.size(0)
                total_used += count
                if count > 0:
                    block_index[b,h,m,:count] = sel
    block_index = block_index.sort(dim=-1).values

    total_possible = batch_size * num_heads * num_M_blocks * (num_N_blocks + 1) // 2
    sparsity_ratio = total_used / total_possible if total_possible > 0 else 0.0

    from my_utils import Logger
    Logger.log(f"total_used:{total_used}, possible:{total_possible}")
    return block_index.int(), sparsity_ratio


def _build_block_index_difference_aware(
    query: torch.Tensor,
    key: torch.Tensor,
    block_size_M: int = 128,
    block_size_N: int = 128,
    theta: float = 0.5,
    sm_scale: float = None,
    top_k: int = float('inf'),
):
    """
    Selects key blocks whose score difference from the max is <= theta.
    Ensures the highest-scoring block is always included.
    Clips to top_k if specified.
    Returns block indices and sparsity ratio.
    """
    batch_size, num_heads, context_size, head_dim = query.shape
    num_M_blocks = (context_size + block_size_M - 1) // block_size_M
    num_N_blocks = (context_size + block_size_N - 1) // block_size_N

    query_pool = query.view(batch_size, num_heads, num_M_blocks, block_size_M, head_dim).mean(dim=-2) * sm_scale
    key_pool = key.view(batch_size, num_heads, num_N_blocks, block_size_N, head_dim).mean(dim=-2)

    arange_M = torch.arange(num_M_blocks, device=query.device) * block_size_M
    arange_N = torch.arange(num_N_blocks, device=query.device) * block_size_N
    p_pool = torch.einsum('bhmk,bhnk->bhmn', query_pool, key_pool)
    p_pool = p_pool.masked_fill(arange_M[None,None,:,None] < arange_N[None,None,None,:], float('-inf'))

    # Compute differences
    max_scores, _ = p_pool.max(dim=-1, keepdim=True)
    diff = max_scores - p_pool
    valid = diff <= theta

    # Always include the max block
    max_idx = p_pool.argmax(dim=-1)
    b_idx = torch.arange(batch_size, device=query.device)[:,None,None]
    h_idx = torch.arange(num_heads, device=query.device)[None,:,None]
    m_idx = torch.arange(num_M_blocks, device=query.device)[None,None,:]
    valid[b_idx,h_idx,m_idx,max_idx] = True

    # Clip by top_k if needed
    if top_k < num_N_blocks:
        scores = p_pool.masked_fill(~valid, float('-inf'))
        topk_idx = scores.topk(min(top_k,num_N_blocks), dim=-1).indices
        valid = torch.zeros_like(valid)
        valid.scatter_(-1, topk_idx, True)

    # Pack indices
    counts = valid.sum(dim=-1)
    max_blocks = min(counts.max().item(), int(top_k))
    block_index = torch.zeros(batch_size, num_heads, num_M_blocks, max_blocks, dtype=torch.int64, device=query.device)
    total_used = 0
    for b in range(batch_size):
        for h in range(num_heads):
            for m in range(num_M_blocks):
                sel = torch.arange(num_N_blocks, device=query.device)[valid[b,h,m]]
                count = sel.size(0)
                total_used += count
                if count > 0:
                    block_index[b,h,m,:min(count,max_blocks)] = sel[:max_blocks]
    block_index = block_index.sort(dim=-1).values

    total_possible = batch_size * num_heads * num_M_blocks * (num_N_blocks + 1) // 2
    sparsity_ratio = total_used / total_possible if total_possible > 0 else 0.0
    return block_index.int(), sparsity_ratio

# ------------------------------
# High-level wrapper for block-sparse attention
# ------------------------------
def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
    mode: str = 'top_k',
    cdf_threshold: float = 0.95,
    theta: float = 0.5,
    return_sparsity_ratio: bool = False,
):
    """
    Computes block-sparse self-attention using one of three selection modes:
      - 'top_k': select fixed number of blocks
      - 'top_cdf': select blocks until cumulative probability exceeds threshold
      - 'top_difference_aware': select blocks within difference threshold
    Optionally returns the sparsity ratio.
    """
    B, H, L, D = query.shape
    pad_len = (-L) % block_size_M
    # Pad sequences to align with block boundaries
    if pad_len > 0:
        query = torch.nn.functional.pad(query, (0,0,0,pad_len))
        key = torch.nn.functional.pad(key, (0,0,0,pad_len))
        value = torch.nn.functional.pad(value, (0,0,0,pad_len))
    seqlens = torch.full((B,), L, dtype=torch.int32, device=query.device)
    sm_scale = D ** -0.5

    # Build block index
    if mode == 'top_k':
        idx, ratio = _build_block_index(query, key, top_k, block_size_M, block_size_N)
    elif mode == 'top_cdf':
        idx, ratio = _build_block_index_top_cdf(query, key, cdf_threshold, block_size_M, block_size_N, sm_scale)
    elif mode == 'top_difference_aware':
        idx, ratio = _build_block_index_difference_aware(query, key, block_size_M, block_size_N, theta, sm_scale, top_k)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'top_k', 'top_cdf', or 'top_difference_aware'.")

    # Execute Triton kernel
    output = _triton_block_sparse_attention(query, key, value, seqlens, idx, sm_scale, block_size_M, block_size_N)
    # Remove padding
    output = output[..., :L, :]
    return (output, ratio) if return_sparsity_ratio else output

# ------------------------------
# Example usage and benchmarking
# ------------------------------
if __name__ == "__main__":
    # Configure logger path
    Logger.set_log_file_path(f"log/test_block/{time.time()}.log")

    # Test settings
    B, H, N, D = 1, 1, 128*1024, 128
    block_M, block_N = 128, 128
    runs = 2
    device = "cuda:0"
    dtype = torch.bfloat16

    # Random inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous() * 2
    k = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous() * 2
    v = torch.randn(B, H, N, D, device=device, dtype=dtype).contiguous() / 2 
    
    # from my_utils.ops_util import load_data
    # q,k,v = load_data(context_length = 64*1024)
    # print(f"q shape:{q.shape}")
    # print(f"k shape:{k.shape}")
    # print(f"v shape:{v.shape}")
    from .flash_attn_triton import triton_flash_attention
    
    # Benchmark and compare modes against full and FlashAttention
    for i in range(runs):
        
        # Full attention
        torch.cuda.synchronize()
        start_full = time.time()
        out_full = triton_flash_attention(q, k, v)
        torch.cuda.synchronize()
        time_full = (time.time() - start_full) * 1000
        Logger.log(f"Run {i}, full_attention: {time_full:.2f} ms")

        # Block-sparse top_k
        torch.cuda.synchronize()
        start = time.time()
        out_k, sp_k = block_sparse_attention(q, k, v, top_k=128, mode='top_k', return_sparsity_ratio=True)
        torch.cuda.synchronize()
        time_k = (time.time() - start) * 1000
        Logger.log(f"Run {i}, block_sparse (top_k): {time_k:.2f} ms, sparsity={sp_k:.4f}")
        diff_k = torch.max((out_full - out_k).abs()).item()
        Logger.log(f"Run {i}, max diff (full vs top_k): {diff_k:.6f}")
        
        diff_k = torch.mean((out_full - out_k).abs()).item()
        Logger.log(f"Run {i}, mean diff (full vs top_k): {diff_k:.6f}")

        # Block-sparse top_cdf
        torch.cuda.synchronize()
        start = time.time()
        out_cdf, sp_cdf = block_sparse_attention(q, k, v, top_k=128, mode='top_cdf', cdf_threshold=0.95, return_sparsity_ratio=True)
        torch.cuda.synchronize()
        time_cdf = (time.time() - start) * 1000
        Logger.log(f"Run {i}, block_sparse (top_cdf): {time_cdf:.2f} ms, sparsity={sp_cdf:.4f}")
        diff_cdf = torch.max((out_full - out_cdf).abs()).item()
        Logger.log(f"Run {i}, max diff (full vs top_cdf): {diff_cdf:.6f}")
        
        diff_cdf = torch.mean((out_full - out_cdf).abs()).item()
        Logger.log(f"Run {i}, mean diff (full vs top_cdf): {diff_cdf:.6f}")
        
        # Block-sparse top_difference_aware
        torch.cuda.synchronize()
        start = time.time()
        out_diff, sp_diff = block_sparse_attention(q, k, v, top_k=128, mode='top_difference_aware', theta=10, return_sparsity_ratio=True)
        torch.cuda.synchronize()
        time_diff = (time.time() - start) * 1000
        Logger.log(f"Run {i}, block_sparse (top_difference_aware): {time_diff:.2f} ms, sparsity={sp_diff:.4f}")
        diff_diff = torch.max((out_full - out_diff).abs()).item()
        Logger.log(f"Run {i}, max diff (full vs top_difference_aware): {diff_diff:.6f}")

        diff_diff = torch.mean((out_full - out_diff).abs()).item()
        Logger.log(f"Run {i}, mean diff (full vs top_difference_aware): {diff_diff:.6f}")
    Logger.log("Benchmark complete.")
