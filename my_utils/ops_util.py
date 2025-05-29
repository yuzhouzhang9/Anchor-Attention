import torch
import os
from .my_log import Logger


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value head tensors to match the number of query heads.

    Args:
        hidden_states (torch.Tensor): Tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        n_rep (int): Number of repetitions needed to match query heads.

    Returns:
        torch.Tensor: Expanded tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim).
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        # No repetition needed
        return hidden_states
    # Insert a repetition dimension, expand, then reshape
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    # Merge kv heads and repetitions
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def load_data(
    path: str = "tensor",
    layer_idx: int = 1,
    context_length: int = 1024,
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and prepare query, key, and value tensors from disk.

    Constructs file paths based on layer index and context length, loads the .pt files,
    casts them to the desired dtype, transposes to (batch, heads, seq, dim),
    and repeats key/value heads if necessary to match query head count.

    Args:
        path (str): Base directory where tensor folders reside.
        layer_idx (int): Subdirectory for the specific transformer layer.
        context_length (int): Subdirectory for the sequence length.
        device (torch.device): Device for tensor allocation.
        dtype (torch.dtype): Desired dtype of tensors.

    Returns:
        tuple: (Q, K, V) tensors ready for attention computation.
    """
    # Build the directory path for this layer and context length
    dir_path = os.path.join(path, str(layer_idx), str(context_length))
    # File names for query, key, and value states
    query_file = os.path.join(dir_path, "query_states.pt")
    key_file = os.path.join(dir_path, "key_states.pt")
    value_file = os.path.join(dir_path, "value_states.pt")

    # Load tensors, convert dtype, and transpose from (batch, seq, heads, dim) to (batch, heads, seq, dim)
    Q = torch.load(query_file, map_location=device, weights_only=True)
    Q = Q.to(dtype).transpose(1, 2)
    K = torch.load(key_file, map_location=device, weights_only=True)
    K = K.to(dtype).transpose(1, 2)
    V = torch.load(value_file, map_location=device, weights_only=True)
    V = V.to(dtype).transpose(1, 2)

    # If the number of key/value heads differs, repeat them to match the query heads
    num_query_heads = Q.size(1)
    num_key_heads = K.size(1)
    num_value_heads = V.size(1)

    if num_query_heads % num_key_heads != 0:
        raise ValueError("Query heads must be a multiple of key heads.")
    rep_factor_k = num_query_heads // num_key_heads
    if rep_factor_k > 1:
        K = repeat_kv(K, rep_factor_k).contiguous()

    if num_query_heads % num_value_heads != 0:
        raise ValueError("Query heads must be a multiple of value heads.")
    rep_factor_v = num_query_heads // num_value_heads
    if rep_factor_v > 1:
        V = repeat_kv(V, rep_factor_v).contiguous()

    return Q, K, V


def get_deff(
    x: torch.Tensor,
    y: torch.Tensor,
    verbose: bool = True,
    strict_threshold: float = 1e-3,
    loose_abs_threshold: float = 1e-4,
    loose_rel_threshold: float = 0.3,
    relative_error: bool = True,
    cosine_similarity: bool = True,
    top_k_diff: int = 1,
    sample_region: tuple[slice, ...] = None,
    log_histogram: bool = False,
    histogram_bins: int = 10
) -> tuple[torch.Tensor, float, float, dict]:
    """
    Compute various difference metrics between two tensors.

    Supports strict and loose equality thresholds, relative errors,
    cosine similarity, and histogram logging for absolute differences.

    Args:
        x (torch.Tensor): Reference tensor.
        y (torch.Tensor): Comparison tensor.
        verbose (bool): If True, logs detailed metrics via Logger.
        strict_threshold (float): Max absolute difference for strict equality.
        loose_abs_threshold (float): Max absolute diff for loose equality.
        loose_rel_threshold (float): Max relative diff (%) for loose equality.
        relative_error (bool): Compute mean relative error.
        cosine_similarity (bool): Compute cosine similarity between flattened tensors.
        top_k_diff (int): Number of largest differences to display (unused).
        sample_region (tuple of slices): Subregion to compare.
        log_histogram (bool): If True, log histogram of abs differences.
        histogram_bins (int): Number of bins for histogram.

    Returns:
        tuple: (mean_abs_error, strict_equal_ratio, loose_equal_ratio, metrics_dict).
    """
    # Validate shape and dtype
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {x.shape}, y {y.shape}")
    if x.dtype != y.dtype:
        Logger.log(f"Warning: dtypes differ x:{x.dtype}, y:{y.dtype}")
    if x.device != y.device:
        Logger.log(f"Warning: devices differ x:{x.device}, y:{y.device}")

    # Optionally select a subregion
    if sample_region is not None:
        x = x[sample_region]
        y = y[sample_region]

    # Absolute differences
    abs_diff = (x - y).abs()
    mean_abs_error = abs_diff.mean()

    # Strict equality ratio
    strict_mask = abs_diff <= strict_threshold
    strict_ratio = strict_mask.sum().item() / strict_mask.numel()

    # Loose equality ratio (abs or relative)
    tiny_mask = abs_diff < loose_abs_threshold
    rel_mask = torch.where(
        x >= 0,
        (y >= x * (1 - loose_rel_threshold)) & (y <= x * (1 + loose_rel_threshold)),
        (y <= x * (1 - loose_rel_threshold)) & (y >= x * (1 + loose_rel_threshold))
    )
    loose_mask = tiny_mask | rel_mask
    loose_ratio = loose_mask.sum().item() / loose_mask.numel()

    # Build metrics dictionary
    metrics: dict = {}

    # Absolute error distribution
    abs_thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    metrics['abs_error_ratios'] = {
        f'<= {th}': float((abs_diff <= th).sum().item()) / abs_diff.numel()
        for th in abs_thresholds
    }

    # Relative error distribution
    rel_diff = abs_diff / (x.abs() + 1e-8)
    rel_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    metrics['rel_error_ratios'] = {
        f'<= {int(th*100)}%': float((rel_diff <= th).sum().item()) / rel_diff.numel()
        for th in rel_thresholds
    }

    if relative_error:
        metrics['mean_relative_error'] = float(rel_diff.mean().item())

    if cosine_similarity:
        # Flatten and compute cosine similarity
        x_flat = x.contiguous().view(-1).float()
        y_flat = y.contiguous().view(-1).float()
        cos_sim = torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=0)
        metrics['cosine_similarity'] = float(cos_sim.item())

    # Optionally log detailed info
    if verbose:
        Logger.log(f"Mean absolute error: {mean_abs_error.item():.3e}")
        Logger.log(f"Strict equality ratio: {strict_ratio*100:.2f}%")
        Logger.log(f"Loose equality ratio: {loose_ratio*100:.2f}%")
        Logger.log(f"Metrics: {metrics}")

    return mean_abs_error, strict_ratio, loose_ratio, metrics


def get_cdf_fig(
    reference: torch.Tensor,
    comparisons: list[tuple[torch.Tensor, str]],
    save_path: str,
) -> None:
    """
    Generate and save a smooth CDF plot comparing reference tensor differences.

    Args:
        reference (torch.Tensor): Base tensor for comparison.
        comparisons (list of (tensor, label)): List of tensors and labels to compare.
        save_path (str): File path to save the resulting figure.
    """
    from figures.fig_scripts.fig_cdf import plot_smooth_cdf

    data = []
    # Helper to compute the percentage of elements within a relative threshold
    def relative_ratio(a: torch.Tensor, b: torch.Tensor, threshold: float) -> float:
        assert a.shape == b.shape, "Tensor shapes must match"
        diff = (a - b).abs()
        mask = torch.where(
            a >= 0,
            (b >= a * (1 - threshold)) & (b <= a * (1 + threshold)),
            (b <= a * (1 - threshold)) & (b >= a * (1 + threshold))
        )
        return float(mask.sum().item()) / mask.numel()

    # Build CDF data for each comparison
    for tensor, label in comparisons:
        thresholds = [i * 0.01 for i in range(1, 100)]
        ratios = [relative_ratio(reference, tensor, t) for t in thresholds]
        data.append((thresholds, ratios, label))

    # Delegate to the plotting utility
    plot_smooth_cdf(data, save_path=save_path)