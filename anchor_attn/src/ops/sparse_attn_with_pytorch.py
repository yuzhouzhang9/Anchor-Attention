import torch
import math

def sparse_attn_topcdf_with_torch(
    q,
    k,
    v,
    softmax_scale=None,
    cdf_threshold: float = 0.9,  # CDF threshold (e.g., 0.9 for 90%)
    fullkv: bool = False,
    return_computational_ratio: bool = False,
):
    """
    Sparse attention with top-CDF strategy, selecting positions based on cumulative softmax weights.

    Args:
        q: Query tensor, shape (batch_size, num_heads, q_len, head_dim).
        k: Key tensor, shape (batch_size, num_heads, seq_len, head_dim).
        v: Value tensor, shape (batch_size, num_heads, seq_len, head_dim).
        softmax_scale: Optional scaling factor for softmax, defaults to 1/sqrt(head_dim).
        cdf_threshold: CDF threshold for selecting top weights (default: 0.9).
        fullkv: If True, use full key-value pairs; if False, use top-CDF strategy.
        return_computational_ratio: If True, return (output, sparsity_ratio); else return output.

    Returns:
        output: Attention output tensor, shape (batch_size, num_heads, q_len, head_dim).
        sparsity_ratio: Optional, returned if return_computational_ratio is True.
    """
    Q = q
    K = k
    V = v
    
    batch_size, num_heads, q_len, head_dim = Q.shape
    seq_len = K.shape[2]
    
    chunk_size = 128
    scale = softmax_scale if softmax_scale is not None else 1 / math.sqrt(head_dim)
    attn_res = torch.zeros(batch_size, num_heads, q_len, head_dim, device=Q.device, dtype=Q.dtype)
    
    total_elements = 0
    sparse_elements = 0
    
    for start_idx in range(0, q_len, chunk_size):
        end_idx = min(start_idx + chunk_size, q_len)
        chunk_query_states = Q[:, :, start_idx:end_idx, :]
        chunk_attention_score = torch.matmul(chunk_query_states, K.transpose(-1, -2)) * scale
        
        # Apply causal mask
        chunk_mask = torch.triu(
            torch.ones(
                end_idx - start_idx,
                seq_len,
                device=chunk_attention_score.device,
                dtype=q.dtype
            ),
            diagonal=start_idx + 1
        )
        chunk_mask = chunk_mask.masked_fill(chunk_mask == 1, float('-inf'))  # Invalid positions are -inf
        valid_mask = chunk_mask == 0  # Valid positions where mask is 0
        chunk_attention_score = chunk_attention_score + chunk_mask.unsqueeze(0).unsqueeze(0)
        
        if not fullkv:
            # Compute softmax to get attention weights
            chunk_attention_weights = torch.softmax(chunk_attention_score, dim=-1).to(torch.float)
            
            # Sort chunk_attention_weights in descending order along the last dimension
            sorted_weights, indices = torch.sort(chunk_attention_weights, dim=-1, descending=True)

            # Compute cumulative sum and normalize to obtain the CDF
            cumsum_weights = torch.cumsum(sorted_weights, dim=-1)
            cdf = cumsum_weights / cumsum_weights[..., -1, None]
            mask = cdf >= cdf_threshold
            weight_pos = torch.argmax(mask.int(), dim=-1, keepdim=True)
            weight_threshold = torch.gather(sorted_weights, dim=-1, index=weight_pos)
            topcdf_mask = (chunk_attention_weights >= weight_threshold) & valid_mask.unsqueeze(0).unsqueeze(0)
            total_elements += valid_mask.sum().item() * batch_size * num_heads
            sparse_elements += min(total_elements, topcdf_mask.sum().item())
            
            # Apply the top-weight mask to attention scores
            chunk_attention_score = torch.where(
                topcdf_mask, chunk_attention_score, torch.full_like(chunk_attention_score, float('-inf'))
            )
            
            # Recompute softmax for selected positions
            chunk_attention_weights = torch.softmax(chunk_attention_score, dim=-1)
        else:
            # Full KV: compute softmax directly
            chunk_attention_weights = torch.softmax(chunk_attention_score, dim=-1)
            total_elements += valid_mask.sum().item() * batch_size * num_heads
            sparse_elements += valid_mask.sum().item() * batch_size * num_heads  # All valid positions are used
        
        chunk_output = torch.matmul(chunk_attention_weights, V)
        attn_res[:, :, start_idx:end_idx, :] = chunk_output
    
    sparsity_ratio = sparse_elements / total_elements if total_elements > 0 else 0.0
    
    if return_computational_ratio:
        return attn_res, sparsity_ratio
    
    return attn_res


def sparse_attn_topk_with_torch(
    q,
    k,
    v,
    softmax_scale=None,
    top_K: int = 1024,
    fullkv: bool = False,
    return_computational_ratio: bool = False,
):
    Q = q
    K = k
    V = v
    # print(f"Inside sparse_attn_topk_with_torch: top_K={top_K}")
    batch_size, num_heads, q_len, head_dim = Q.shape
    seq_len = K.shape[2]
    
    chunk_size = 256
    scale = softmax_scale if softmax_scale is not None else 1 / math.sqrt(head_dim)
    attn_res = torch.zeros(batch_size, num_heads, q_len, head_dim, device=Q.device, dtype=Q.dtype)
    
    total_elements = 0
    sparse_elements = 0
    
    for start_idx in range(0, q_len, chunk_size):
        end_idx = min(start_idx + chunk_size, q_len)
        chunk_query_states = Q[:, :, start_idx:end_idx, :]
        chunk_attention_score = torch.matmul(chunk_query_states, K.transpose(-1, -2)) * scale
        
        # Apply causal mask
        chunk_mask = torch.triu(
            torch.ones(
                end_idx - start_idx,
                seq_len,
                device=chunk_attention_score.device,
                dtype=q.dtype
            ),
            diagonal=start_idx + 1
        )
        chunk_mask = chunk_mask.masked_fill(chunk_mask == 1, float('-inf'))  # Invalid positions are -inf
        valid_mask = chunk_mask == 0  # Valid positions where mask is 0
        chunk_attention_score = chunk_attention_score + chunk_mask.unsqueeze(0).unsqueeze(0)
        
        if not fullkv:
            # Apply top-k strategy
            # Select top-K scores and their indices along the sequence length dimension
            topk_scores, topk_indices = torch.topk(chunk_attention_score, k=top_K, dim=-1, largest=True)
            
            # Create a mask for top-k positions
            topk_mask = torch.zeros_like(chunk_attention_score, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, 1)
            
            # Apply top-k mask to attention scores (non-top-k positions set to -inf)
            chunk_attention_score = torch.where(
                topk_mask, chunk_attention_score, torch.full_like(chunk_attention_score, float('-inf'))
            )
            
            # Update sparsity metrics
            total_elements += valid_mask.sum().item() * batch_size * num_heads  # Count valid positions
            sparse_elements += min(topk_mask.sum().item(), valid_mask.sum().item() * batch_size * num_heads) # Count selected top-k positions
            from my_utils import Logger
        
        chunk_attention_weights = torch.softmax(chunk_attention_score, dim=-1)
        chunk_output = torch.matmul(chunk_attention_weights, V)
        attn_res[:, :, start_idx:end_idx, :] = chunk_output
    
    sparsity_ratio = sparse_elements / total_elements if total_elements > 0 else 0.0
    if return_computational_ratio:
        return attn_res, sparsity_ratio
    
    return attn_res


def sparse_attn_with_torch(
    q,
    k,
    v,
    softmax_scale=None,
    diff_anchor=12,
    fullkv: bool = False,
    return_computational_ratio: bool = False,
):
    Q = q
    K = k
    V = v
    
    batch_size, num_heads, q_len, head_dim = Q.shape
    seq_len = K.shape[2]
    
    chunk_size = 256
    scale = softmax_scale if softmax_scale is not None else 1 / math.sqrt(head_dim)
    attn_res = torch.zeros(batch_size, num_heads, q_len, head_dim, device=Q.device, dtype=Q.dtype)
    
    total_elements = 0
    sparse_elements = 0
    
    for start_idx in range(0, q_len, chunk_size):
        end_idx = min(start_idx + chunk_size, q_len)
        chunk_query_states = Q[:, :, start_idx:end_idx, :]
        chunk_attention_score = torch.matmul(chunk_query_states, K.transpose(-1, -2)) * scale
        
        # Apply causal mask
        chunk_mask = torch.triu(
            torch.ones(
                end_idx - start_idx,
                seq_len,
                device=chunk_attention_score.device,
                dtype=q.dtype
            ),
            diagonal=start_idx + 1
        )
        chunk_mask = chunk_mask.masked_fill(chunk_mask == 1, float('-inf'))  # Invalid positions are -inf
        valid_mask = chunk_mask == 0  # Positions where mask is 0 are valid
        chunk_attention_score = chunk_attention_score + chunk_mask.unsqueeze(0).unsqueeze(0)
        
        if not fullkv:
            # Identify maximum score and its index
            chunk_max_value, chunk_max_idx = torch.max(chunk_attention_score, dim=-1, keepdim=True)
            # Create sparse mask for positions below (max - diff_anchor)
            sparse_mask = ((chunk_attention_score + diff_anchor) < chunk_max_value) & valid_mask
            sparse_elements += sparse_mask.sum().item()
            total_elements += valid_mask.sum().item() * batch_size * num_heads  # Count valid positions
            
            # Mask out sparse positions by setting scores to -inf
            chunk_attention_score = torch.where(
                sparse_mask,
                torch.tensor(float('-inf'), device=chunk_attention_score.device, dtype=chunk_attention_score.dtype),
                chunk_attention_score
            )
        
        chunk_attention_weights = torch.softmax(chunk_attention_score, dim=-1)
        chunk_output = torch.matmul(chunk_attention_weights, V)
        attn_res[:, :, start_idx:end_idx, :] = chunk_output
    
    sparsity_ratio = (total_elements - sparse_elements) / total_elements if total_elements > 0 else 1.0
    
    if return_computational_ratio:
        return attn_res, sparsity_ratio
    
    return attn_res

def test_attention_methods(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    diff_anchors: list = [0, 3, 6, 9, 12, 15],
    topk_values: list = [512, 1024, 2048],
    cdf_thresholds: list = [0.8, 0.9, 0.95],
    num_runs: int = 3,
):
    """
    Test and compare different sparse attention methods.

    Args:
        q: Query tensor, shape (batch_size, num_heads, seq_len, head_dim).
        k: Key tensor, shape (batch_size, num_heads, seq_len, head_dim).
        v: Value tensor, shape (batch_size, num_heads, seq_len, head_dim).
        softmax_scale: Scaling factor for softmax.
        diff_anchors: List of diff_anchor values for sparse_attn_with_torch.
        topk_values: List of K values for sparse_attn_topk_with_torch.
        cdf_thresholds: List of CDF thresholds for sparse_attn_topcdf_with_torch.
        num_runs: Number of runs to average timing results.

    Returns:
        dict: Results containing outputs, times, sparsity ratios, and differences.
    """
    Logger.log("\n=== Testing Sparse Attention Methods ===")
    Logger.log(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    Logger.log(f"Softmax scale: {softmax_scale:.4f}")

    results = {}

    # Helper function to time a method
    def time_method(method, *args, **kwargs):
        times = []
        output = None
        sparsity_ratio = None
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            result = method(*args, **kwargs)
            torch.cuda.synchronize()
            times.append((time.time() - start_time) * 1000)  # Time in ms
            if isinstance(result, tuple):
                output, sparsity_ratio = result
            else:
                output = result
        avg_time = sum(times) / len(times)
        return output, times, avg_time, sparsity_ratio

    # Compute full attention as reference
    Logger.log("\nRunning Full Attention (Reference)...")
    full_method = lambda: sparse_attn_with_torch(q, k, v, softmax_scale, fullkv=True)
    full_output, full_times, full_avg_time, _ = time_method(full_method)
    Logger.log(f"Full Attention:")
    Logger.log(f"  Times: {[f'{t:.2f}' for t in full_times]} ms")
    Logger.log(f"  Average Time: {full_avg_time:.2f} ms")
    Logger.log(f"  Output Mean: {full_output.abs().mean().item():.4f}")
    results["full"] = {"output": full_output, "times": full_times, "avg_time": full_avg_time}

    # Test sparse_attn_with_torch
    Logger.log("\nTesting sparse_attn_with_torch...")
    for diff_anchor in diff_anchors:
        Logger.log(f"\ndiff_anchor={diff_anchor}")
        sparse_method = lambda: sparse_attn_with_torch(
            q, k, v, softmax_scale, diff_anchor=diff_anchor, fullkv=False, return_computational_ratio=True
        )
        output, times, avg_time, sparsity_ratio = time_method(sparse_method)
        diff_full = (full_output - output).abs().mean().item()
        Logger.log(f"  Times: {[f'{t:.2f}' for t in times]} ms")
        Logger.log(f"  Average Time: {avg_time:.2f} ms")
        Logger.log(f"  Sparsity Ratio: {sparsity_ratio:.4f}")
        Logger.log(f"  Output Mean: {output.abs().mean().item():.4f}")
        Logger.log(f"  Diff with Full Attention: {diff_full:.4f}")
        Logger.log(f"  sparsity_ratio: {sparsity_ratio:.4f}")
        get_deff(full_output, output)
        results[f"sparse_diff_{diff_anchor}"] = {
            "output": output,
            "times": times,
            "avg_time": avg_time,
            "sparsity_ratio": sparsity_ratio,
            "diff_full": diff_full
        }

    # # Test sparse_attn_topk_with_torch
    Logger.log("\nTesting sparse_attn_topk_with_torch...")
    for top_K in topk_values:
        Logger.log(f"\nK={top_K}")
        topk_method = lambda: sparse_attn_topk_with_torch(
            q, k, v, softmax_scale, top_K=top_K, fullkv=False, return_computational_ratio=True
        )
        output, times, avg_time, sparsity_ratio = time_method(topk_method)
        diff_full = (full_output - output).abs().mean().item()
        Logger.log(f"  Times: {[f'{t:.2f}' for t in times]} ms")
        Logger.log(f"  Average Time: {avg_time:.2f} ms")
        Logger.log(f"  Sparsity Ratio: {sparsity_ratio:.4f}")
        Logger.log(f"  Output Mean: {output.abs().mean().item():.4f}")
        Logger.log(f"  Diff with Full Attention: {diff_full:.4f}")
        Logger.log(f"  sparsity_ratio: {sparsity_ratio:.4f}")
        get_deff(full_output, output)
        results[f"topk_{top_K}"] = {
            "output": output,
            "times": times,
            "avg_time": avg_time,
            "sparsity_ratio": sparsity_ratio,
            "diff_full": diff_full
        }

    # Test sparse_attn_topcdf_with_torch
    Logger.log("\nTesting sparse_attn_topcdf_with_torch...")
    for cdf_threshold in cdf_thresholds:
        Logger.log(f"\ncdf_threshold={cdf_threshold}")
        topcdf_method = lambda: sparse_attn_topcdf_with_torch(
            q, k, v, softmax_scale, cdf_threshold=cdf_threshold, fullkv=False, return_computational_ratio=True
        )
        output, times, avg_time, sparsity_ratio = time_method(topcdf_method)
        diff_full = (full_output - output).abs().mean().item()
        Logger.log(f"  Times: {[f'{t:.2f}' for t in times]} ms")
        Logger.log(f"  Average Time: {avg_time:.2f} ms")
        Logger.log(f"  Sparsity Ratio: {sparsity_ratio:.4f}")
        Logger.log(f"  Output Mean: {output.abs().mean().item():.4f}")
        Logger.log(f"  Diff with Full Attention: {diff_full:.4f}")
        Logger.log(f"  sparsity_ratio: {sparsity_ratio:.4f}")
        get_deff(full_output, output)
        results[f"topcdf_{cdf_threshold}"] = {
            "output": output,
            "times": times,
            "avg_time": avg_time,
            "sparsity_ratio": sparsity_ratio,
            "diff_full": diff_full
        }

    return results


if __name__ == "__main__":
    from my_utils.ops_util import get_deff
    from my_utils import Logger
    import time
    # Test parameters
    batch_size = 1
    num_heads = 32
    seq_len = 16 * 1024
    head_dim = 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # or torch.float16
    
    # Create random input tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device) * 2
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device) * 2
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device) / 2
    
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Configure Logger
    Logger.set_log_file_path(f"sparse_attn_test_{time.time()}.log")
    
    # Run tests
    results = test_attention_methods(
        q, k, v,
        softmax_scale=softmax_scale,
        diff_anchors=[0, 3, 6, 9, 12, 15],
        topk_values=[512, 1024, 2048],
        cdf_thresholds=[0.8, 0.9, 0.95],
        num_runs=3
    )
    
    # Summarize results
    Logger.log("\n=== Summary ===")
    Logger.log("Method | Avg Time (ms) | Sparsity Ratio | Diff with Full")
    Logger.log("-" * 50)
    for key, result in results.items():
        if key == "full":
            Logger.log(f"{key:20} | {result['avg_time']:.2f} | 1.0000 | 0.0000")
        else:
            Logger.log(f"{key:20} | {result['avg_time']:.2f} | {result['sparsity_ratio']:.4f} | {result['diff_full']:.4f}")