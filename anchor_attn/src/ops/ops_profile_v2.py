import torch
import time
from .flash_attn_triton import triton_flash_attention
from .anchor_attn import anchor_attn
from .anchor_attn_pytorch import anchor_attn_torch_ops
from .sparse_attn_with_pytorch import (
    sparse_attn_with_torch,
    sparse_attn_topcdf_with_torch,
    sparse_attn_topk_with_torch,
)
from .streaming_llm_attention import streaming_llm_attention
from .vertical_slash_attention import vertical_slash_attention
from .block_sparse_flash_attention import block_sparse_attention
from .flex_prefill_attention import flex_prefill_attention
from my_utils.ops_util import load_data, get_deff
from my_utils import Logger

# Unified config dictionary containing parameter settings for all attention methods
CONFIG = {
    "recall_thresholds": [0.01, 0.005],
    "output_dir": "./ops_tensors",
    "anchor_attention": {
        # "differences": [9, 10, 11, 12, 13, 14, 15, 20, 1000]  # Anchor Attention difference parameter
        # "differences": [9,11,15]
        # "differences": [9]
        "differences": [9, 10, 11, 12, 13, 14, 15, 16, 17],
    },
    "sparse_topcdf_attention": {
        "cdf_thresholds": [0.7, 0.8, 0.85, 0.9, 0.95]  # Sparse TopCDF Attention cdf_threshold parameter
    },
    "sparse_topk_attention": {
        "top_k_values": [512, 1024, 2048, 4096]  # Sparse TopK Attention top_K parameter
    },
    "sparse_attention": {
        "diff_anchors": [1, 2, 6, 9, 10, 11, 12, 13, 14, 20]  # Sparse Attention diff_anchor parameter
    },
    "vertical_slash_attention": {
        "sizes": [
            {"vertical_size": 512, "slash_size": 1024},
            {"vertical_size": 1024, "slash_size": 2048},
            {"vertical_size": 2048, "slash_size": 4096},
            {"vertical_size": 4096, "slash_size": 8192},
            {"vertical_size": 4096 * 2, "slash_size": 8192 * 2},
        ]  # Vertical Slash Attention vertical_size & slash_size parameters
    },
    "flex_prefill_attention": {
        "params": [
            {"gamma": 0.9, "tau": 0.1},
            {"gamma": 0.91, "tau": 0.1},
            {"gamma": 0.92, "tau": 0.1},
            {"gamma": 0.93, "tau": 0.1},
            {"gamma": 0.94, "tau": 0.1},
            {"gamma": 0.95, "tau": 0.1},
            {"gamma": 0.96, "tau": 0.1},
            {"gamma": 0.97, "tau": 0.1},
            {"gamma": 0.98, "tau": 0.1},
            {"gamma": 0.99, "tau": 0.1},
        ]  # Flex Prefill Attention gamma & tau parameters
    },
    "streaming_llm_attention": {
        "window_configs": [
            {"global_window": 512, "local_window": 256},
            {"global_window": 1024, "local_window": 512},
            {"global_window": 2048, "local_window": 1024},
            {"global_window": 4096, "local_window": 2048},
            {"global_window": 8192, "local_window": 4096},
            {"global_window": 8192 * 2, "local_window": 4096 * 2},
        ]  # Streaming LLM Attention global_window & local_window parameters
    },
    "block_sparse_attention": {
        "top_k_configs": [
            {"top_k": 512, "block_size_M": 64, "block_size_N": 64},
            {"top_k": 1024, "block_size_M": 64, "block_size_N": 64},
            {"top_k": 2048, "block_size_M": 64, "block_size_N": 64},
            {"top_k": 4096, "block_size_M": 64, "block_size_N": 64},
            {"top_k": 8192, "block_size_M": 64, "block_size_N": 64},
        ]  # Block Sparse Attention top_k & block_size parameters
    },
}


def time_attention_method(method, *args, num_runs=1, **kwargs):
    """
    Measure the execution time of an attention method over multiple runs.

    Args:
        method: The attention function to benchmark.
        num_runs: Number of runs to average over (default: 2).
        *args, **kwargs: Arguments to pass to the method.

    Returns:
        tuple: (output of the method, list of execution times in ms, optional sparsity ratio)
    """
    times = []
    output = None
    sparsity_ratio = None
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        result = method(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.time() - start_time) * 1000)
        if isinstance(result, tuple):  # Methods returning (output, sparsity_ratio)
            output, sparsity_ratio = result
        else:
            output = result
    return output, times, sparsity_ratio


def print_timing_results(method_name, times, layer_idx, sparsity_ratio=None):
    """
    Print timing results and optional sparsity ratio for a specific layer.

    Args:
        method_name: Name of the attention method.
        times: List of execution times in ms.
        layer_idx: Index of the layer being benchmarked.
        sparsity_ratio: Optional sparsity ratio to print.
    """
    avg_time = sum(times) / len(times)
    Logger.log(f"Layer {layer_idx} - {method_name}:")
    for i, t in enumerate(times):
        Logger.log(f"  Times: {t:.2f} ms")
    Logger.log(f"  Average: {avg_time:.2f} ms")
    if sparsity_ratio is not None:
        Logger.log(f"  Sparsity Ratio: {sparsity_ratio:.6f}")


def print_diff_results(reference, method_output, method_name, layer_idx, results):
    """
    Print the difference between reference and method output for a specific layer, including sparsity ratio.

    Args:
        reference: Reference output tensor.
        method_output: Output tensor from the method.
        method_name: Name of the attention method.
        layer_idx: Index of the layer being benchmarked.
        results: Dictionary containing results including sparsity ratios.
    """
    method_key = method_name.lower().replace(" ", "_")
    sparsity_ratio = results[method_key]["sparsity_ratio"] if method_key in results else None

    Logger.log(f"Layer {layer_idx} - {method_name} Difference:")
    get_deff(reference, method_output)
    if sparsity_ratio is not None:
        Logger.log(f"  Sparsity Ratio in Comparison: {sparsity_ratio:.6f}")


def load_qkv_data(data_path, layer_idx, context_length, device, dtype, head_id=-1):
    """
    Load Q, K, V tensors for a specific layer.

    Args:
        data_path: Path to the tensor data.
        layer_idx: Index of the layer to load.
        context_length: Length of the context.
        dtype: Data type for the tensors.

    Returns:
        tuple: (Q, K, V) tensors.
    """
    Q, K, V = load_data(
        path=data_path,
        layer_idx=layer_idx,
        context_length=context_length,
        device=device,
        dtype=dtype,
        head_id=head_id,
    )
    return Q, K, V


# Independent test function: Anchor Attention
def test_anchor_attention(Q, K, V, layer_idx, results):
    differences = CONFIG["anchor_attention"]["differences"]

    for difference in differences:
        method_name = f"Anchor Attention (difference={difference})"

        method = lambda: anchor_attn(
            Q, K, V, block_size_M=64, difference=difference, return_computational_ratio=True
        )
        output, times, sparsity_ratio = time_attention_method(method)
        Logger.log(f"output:{output}")

        # Handle output
        if isinstance(output, tuple):
            output, sparsity_ratio = output

        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio,
        }


# Independent test function: Sparse TopCDF Attention
def test_sparse_topcdf_attention(Q, K, V, layer_idx, results):
    cdf_thresholds = CONFIG["sparse_topcdf_attention"]["cdf_thresholds"]

    for cdf in cdf_thresholds:
        method_name = f"Sparse TopCDF Attention (cdf_threshold={cdf})"

        method = lambda: sparse_attn_topcdf_with_torch(
            Q, K, V, cdf_threshold=cdf, fullkv=False, return_computational_ratio=True
        )
        Logger.log(f"Calling sparse_attn_topcdf_with_torch TopCDF={cdf}")
        output, times, sparsity_ratio = time_attention_method(method)

        # Handle output
        if isinstance(output, tuple):
            output, sparsity_ratio = output

        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio,
        }


# Independent test function: Sparse TopK Attention
def test_sparse_topk_attention(Q, K, V, layer_idx, results):
    top_k_values = CONFIG["sparse_topk_attention"]["top_k_values"]

    for k in top_k_values:
        method_name = f"Sparse TopK Attention (top_K={k})"

        method = lambda: sparse_attn_topk_with_torch(
            Q, K, V, top_K=k, fullkv=False, return_computational_ratio=True
        )
        Logger.log(f"Calling sparse_attn_topk_with_torch, top_K={k}")
        output, times, sparsity_ratio = time_attention_method(method)

        # Handle output
        if isinstance(output, tuple):
            output, sparsity_ratio = output

        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio,
        }


# Independent test function: Sparse Attention
def test_sparse_attention(Q, K, V, layer_idx, results):
    diff_anchors = CONFIG["sparse_attention"]["diff_anchors"]

    for diff in diff_anchors:
        method_name = f"Sparse Attention (diff_anchor={diff})"

        method = lambda: sparse_attn_with_torch(
            Q, K, V, diff_anchor=diff, fullkv=False, return_computational_ratio=True
        )
        Logger.log(f"Calling Sparse Attention diff_anchor={diff}")
        output, times, sparsity_ratio = time_attention_method(method)

        # Handle output
        if isinstance(output, tuple):
            output, sparsity_ratio = output

        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio,
        }


# Independent test function: Vertical Slash Attention
def test_vertical_slash_attention(Q, K, V, layer_idx, results):
    sizes = CONFIG["vertical_slash_attention"]["sizes"]

    for size_config in sizes:
        vertical_size = size_config["vertical_size"]
        slash_size = size_config["slash_size"]
        method_name = (
            f"Vertical Slash Attention (vertical_size={vertical_size}, slash_size={slash_size})"
        )

        method = lambda: vertical_slash_attention(
            Q.transpose(1, 2).to(torch.bfloat16),
            K.transpose(1, 2).to(torch.bfloat16),
            V.transpose(1, 2).to(torch.bfloat16),
            vertical_size=vertical_size,
            slash_size=slash_size,
            gqa_interleave=False,
            block_size=128,
        ).transpose(1, 2)

        output, times, sparsity_ratio = time_attention_method(method)

        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": None,
        }


# Independent test function: Flex Prefill Attention
def test_flex_prefill_attention(Q, K, V, layer_idx, results):
    params = CONFIG["flex_prefill_attention"]["params"]

    for param in params:
        gamma = param["gamma"]
        tau = param["tau"]
        method_name = f"Flex Prefill Attention (gamma={gamma}, tau={tau})"

        method = lambda: flex_prefill_attention(
            Q.transpose(1, 2).to(torch.bfloat16),
            K.transpose(1, 2).to(torch.bfloat16),
            V.transpose(1, 2).to(torch.bfloat16),
            gamma=gamma,
            tau=tau,
            min_budget=None,
            max_budget=None,
            gqa_interleave=False,
            block_size=128,
            return_computational_ratio=True,
        )

        output, times, sparsity_ratio = time_attention_method(method)

        # Handle output
        if isinstance(output, tuple):
            output = output[0].transpose(1, 2)
            sparsity_ratio = output[1]
        else:
            output = output.transpose(1, 2)

        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio,
        }


# New test function: Streaming LLM Attention
def test_streaming_llm_attention(Q, K, V, layer_idx, results):
    window_configs = CONFIG["streaming_llm_attention"]["window_configs"]

    for config in window_configs:
        global_window = config["global_window"]
        local_window = config["local_window"]
        method_name = f"Streaming LLM Attention (global_window={global_window}, local_window={local_window})"

        method = lambda: streaming_llm_attention(
            Q.transpose(1, 2).to(torch.bfloat16),
            K.transpose(1, 2).to(torch.bfloat16),
            V.transpose(1, 2).to(torch.bfloat16),
            global_window=global_window,
            local_window=local_window,
            gqa_interleave=False,
        ).transpose(1, 2)

        output, times, sparsity_ratio = time_attention_method(method)

        # Streaming LLM Attention does not return sparsity; keep sparsity_ratio as None
        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": None,
        }


# New test function: Block Sparse Attention
def test_block_sparse_attention(Q, K, V, layer_idx, results):
    top_k_configs = CONFIG["block_sparse_attention"]["top_k_configs"]

    for config in top_k_configs:
        top_k = config["top_k"]
        block_size_M = config["block_size_M"]
        block_size_N = config["block_size_N"]
        method_name = f"Block Sparse Attention (top_k={top_k}, block_size_M={block_size_M}, block_size_N={block_size_N})"

        method = lambda: block_sparse_attention(
            Q.to(torch.bfloat16),
            K.to(torch.bfloat16),
            V.to(torch.bfloat16),
            top_k=top_k,
            block_size_M=block_size_M,
            block_size_N=block_size_N,
        )

        output, times, sparsity_ratio = time_attention_method(method)

        # Block Sparse Attention does not return sparsity; keep sparsity_ratio as None
        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")

        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": None,
        }


def benchmark_attention_methods(Q, K, V, layer_idx, context_length):
    """
    Benchmark various attention methods for a given set of Q, K, V tensors and layer.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        layer_idx: Index of the layer being benchmarked.
        context_length: Length of the context.

    Returns:
        dict: Dictionary containing outputs, times, and sparsity ratios for each method.
    """
    results = {}
    profile = []

    # Flash Attention as baseline
    flash_config = {
        "name": "Flash Attention",
        "method": lambda: triton_flash_attention(
            Q.transpose(1, 2).to(torch.bfloat16),
            K.transpose(1, 2).to(torch.bfloat16),
            V.transpose(1, 2).to(torch.bfloat16),
            causal=True,
        ).transpose(1, 2),
        "sparsity": False,
    }
    output, times, _ = time_attention_method(flash_config["method"])
    print_timing_results(flash_config["name"], times, layer_idx)
    Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")
    results["flash_attention"] = {
        "output": output,
        "times": times,
        "sparsity_ratio": None,
    }
    profile.append(flash_config)

    # Independent test: Anchor Attention
    Logger.log("\n=== Testing Anchor Attention ===")
    test_anchor_attention(Q, K, V, layer_idx, results)
    profile.extend(
        [
            {"name": f"Anchor Attention (difference={d})", "sparsity": True}
            for d in CONFIG["anchor_attention"]["differences"]
        ]
    )

    # Independent test: Flex Prefill Attention
    Logger.log("\n=== Testing Flex Prefill Attention ===")
    test_flex_prefill_attention(Q, K, V, layer_idx, results)
    profile.extend(
        [
            {"name": f"Flex Prefill Attention (gamma={p['gamma']}, tau={p['tau']})", "sparsity": True}
            for p in CONFIG["flex_prefill_attention"]["params"]
        ]
    )

    return results, profile


def get_final_ratio(sparsity_raw_value, method_key):
    """
    Normalize sparsity_ratio to a float or NaN.

    Args:
        sparsity_raw_value: Raw sparsity value (tensor, float, etc.).
        method_key: Method identifier.

    Returns:
        float: Final sparsity value or NaN.
    """
    final_sparsity_value = None

    if sparsity_raw_value is None:
        print(f"Warning: 'sparsity_ratio' not found for method {method_key}. Using NaN.")
        final_sparsity_value = float("nan")
    elif isinstance(sparsity_raw_value, torch.Tensor):
        if sparsity_raw_value.numel() == 1:  # Ensure single element tensor
            final_sparsity_value = sparsity_raw_value.item()
        else:
            print(
                f"Warning: 'sparsity_ratio' for method {method_key} is a multi-element tensor: {sparsity_raw_value}. Using NaN."
            )
            final_sparsity_value = float("nan")
    elif isinstance(sparsity_raw_value, (float, int)):  # Already float or int
        final_sparsity_value = float(sparsity_raw_value)
    else:
        print(
            f"Warning: 'sparsity_ratio' for method {method_key} has unexpected type: {type(sparsity_raw_value)}. Using NaN."
        )
        final_sparsity_value = float("nan")
    return final_sparsity_value


# Record recall and sparsity for each threshold; if flag is False sparsity is omitted
def record_recall_ratios(results, profile, recall_threshold, layer_idx, head_idx, recall_obj):
    flash_key = "flash_attention"
    flash_output = results[flash_key]["output"]
    for config in profile:
        method_name = config["name"]
        method_key = method_name.lower().replace(" ", "_")

        if method_key == flash_key:  # Skip Flash Attention itself
            continue
        if method_key in results:
            ref = results[method_key]["output"]
            diff = torch.abs(ref - flash_output)
            recall_obj.setdefault(method_key, {}).setdefault(layer_idx, {})
            spa_ratio = results[method_key]["sparsity_ratio"]
            spa_ratio = get_final_ratio(spa_ratio, method_key)
            recall_obj[method_key][layer_idx][head_idx] = [
                (diff < recall_threshold).sum().item() / ref.numel(),
                spa_ratio,
            ]


def compare_attention_outputs(results, profile, layer_idx, head_idx):
    """
    Compare the outputs of different attention methods against Flash Attention for a specific layer.

    Args:
        results: Dictionary containing outputs and times for each method.
        profile: List of method configurations used in benchmarking.
        layer_idx: Index of the layer being benchmarked.
    """
    flash_key = "flash_attention"
    if flash_key not in results:
        Logger.log(f"Layer {layer_idx} - No Flash Attention output for comparison.")
        return

    flash_output = results[flash_key]["output"]
    Logger.log(f"Layer {layer_idx} - Output Differences (Compared to Flash Attention):")

    for config in profile:
        method_name = config["name"]
        method_key = method_name.lower().replace(" ", "_")

        if method_key == flash_key:  # Skip Flash Attention itself
            continue

        if method_key in results:
            print_diff_results(
                flash_output, results[method_key]["output"], method_name, layer_idx, results
            )
        else:
            Logger.log(f"Layer {layer_idx} - {method_name} not found in results.")


def save_recall_and_sparsity_data_to_files(recall_obj, base_dir="recall_data_output"):
    """
    Save recall_obj data to files following the specified directory structure.
    recall_obj structure: {context_length: {threshold: {method: {layer_idx: {head_idx: [recall, sparsity]}}}}}

    Output directory structure:
    base_dir/
    ├── {context_length}/
    │   ├── {threshold}/
    │   │   ├── {method_name}_recall.csv
    │   │   ├── {method_name}_sparsity.csv
    │   │   └── ...
    │   └── ...
    └── ...
    """
    import os
    import pandas as pd

    print(f"Base directory for this run: {os.path.abspath(base_dir)}")

    for context_length, cl_data in recall_obj.items():
        cl_str = str(context_length)
        cl_dir_path = os.path.join(base_dir, cl_str)
        os.makedirs(cl_dir_path, exist_ok=True)
        print(f"  Created/Ensured directory: {cl_dir_path}")

        for threshold, th_data in cl_data.items():
            threshold_str = str(threshold).replace(".", "_")
            th_dir_path = os.path.join(cl_dir_path, threshold_str)
            os.makedirs(th_dir_path, exist_ok=True)
            print(f"    Created/Ensured directory: {th_dir_path}")

            for method_name, method_layer_data in th_data.items():
                safe_method_name = "".join(
                    c if c.isalnum() or c in ("_", "-") else "_" for c in method_name
                )

                # --- Prepare data ---
                all_layer_indices = sorted(method_layer_data.keys())
                if not all_layer_indices:
                    print(
                        f"      Skipping {method_name} for CtxLen={context_length}, Thresh={threshold}: No layer data."
                    )
                    continue

                all_head_ids_set = set()
                has_valid_data_for_method = False
                for layer_idx in all_layer_indices:
                    if isinstance(method_layer_data[layer_idx], dict):
                        for head_id, values in method_layer_data[layer_idx].items():
                            if isinstance(values, (list, tuple)) and len(values) >= 1:
                                all_head_ids_set.add(head_id)
                                has_valid_data_for_method = True
                    else:
                        print(
                            f"      Warning: Expected dict for layer {layer_idx} in {method_name}, got {type(method_layer_data[layer_idx])}. Skipping this layer."
                        )

                if not has_valid_data_for_method or not all_head_ids_set:
                    print(
                        f"      Skipping {method_name} for CtxLen={context_length}, Thresh={threshold}: No valid head data found across layers."
                    )
                    continue

                sorted_head_ids = sorted(
                    all_head_ids_set,
                    key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else str(x),
                )

                # Initialize two data lists: one for recall, one for sparsity
                recall_table_data = []
                sparsity_table_data = []

                found_recall_data = False
                found_sparsity_data = False

                for layer_idx in all_layer_indices:
                    recall_row_data = {"Layer_ID": f"Layer_{layer_idx}"}
                    sparsity_row_data = {"Layer_ID": f"Layer_{layer_idx}"}

                    head_data_for_layer = method_layer_data.get(layer_idx, {})

                    for head_id in sorted_head_ids:
                        column_name = f"Head_{head_id}"
                        values = head_data_for_layer.get(head_id)

                        recall_value = float("nan")
                        sparsity_value = float("nan")

                        if isinstance(values, (list, tuple)):
                            if len(values) >= 1:
                                recall_value = values[0]
                                found_recall_data = True
                            if len(values) >= 2:
                                sparsity_value = values[1]
                                found_sparsity_data = True

                        recall_row_data[column_name] = recall_value
                        sparsity_row_data[column_name] = sparsity_value

                    recall_table_data.append(recall_row_data)
                    sparsity_table_data.append(sparsity_row_data)

                # --- Save Recall CSV ---
                if found_recall_data and recall_table_data:
                    recall_csv_filename = f"{safe_method_name}_recall.csv"
                    recall_csv_filepath = os.path.join(th_dir_path, recall_csv_filename)
                    df_recall = pd.DataFrame(recall_table_data)
                    if "Layer_ID" in df_recall.columns:
                        df_recall = df_recall.set_index("Layer_ID")
                    try:
                        df_recall.to_csv(recall_csv_filepath)
                        print(f"      Saved recall data to: {recall_csv_filepath}")
                    except Exception as e:
                        print(f"      Error saving recall data {recall_csv_filepath}: {e}")
                else:
                    print(
                        f"      No recall data to write for {safe_method_name} (CtxLen={context_length}, Thresh={threshold})"
                    )

                # --- Save Sparsity CSV ---
                if found_sparsity_data and sparsity_table_data:
                    sparsity_csv_filename = f"{safe_method_name}_sparsity.csv"
                    sparsity_csv_filepath = os.path.join(th_dir_path, sparsity_csv_filename)
                    df_sparsity = pd.DataFrame(sparsity_table_data)
                    if "Layer_ID" in df_sparsity.columns:
                        df_sparsity = df_sparsity.set_index("Layer_ID")
                    try:
                        df_sparsity.to_csv(sparsity_csv_filepath)
                        print(f"      Saved sparsity data to: {sparsity_csv_filepath}")
                    except Exception as e:
                        print(f"      Error saving sparsity data {sparsity_csv_filepath}: {e}")
                else:
                    print(
                        f"      No sparsity data to write for {safe_method_name} (CtxLen={context_length}, Thresh={threshold})"
                    )


def create_timestamped_directory(parent_dir="."):
    """
    Create a directory named with the current timestamp under the specified parent directory.
    Directory name format: YYYY-MM-DD_HH-MM-SS

    Returns:
        str: Absolute path of the new directory.
    """
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dir_path = os.path.join(parent_dir, timestamp)
    os.makedirs(new_dir_path, exist_ok=True)  # exist_ok=True in case script runs multiple times per second
    return os.path.abspath(new_dir_path)


def benchmark_all_layers():
    """
    Benchmark attention mechanisms for layers 0 to 31 and compare their outputs.
    """
    # Configuration
    Logger.set_log_file_path(f"profile_ops/{time.time()}.log")
    torch.manual_seed(42)
    B, H, N, D = 1, 32, 1024 * 4, 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    data_path = "./tensor"

    recall_obj = {}
    for c in [1, 2, 4, 8, 16, 32, 64, 128]:
        context_length = c * 1024
        recall_obj.setdefault(context_length, {})
        for layer_idx in range(32):
            for head_id in range(32):
                Logger.log(f"\n=== Benchmarking Layer {layer_idx} ===")
                Logger.log(f"\n=== Benchmarking head_id {head_id} ===")

                # Load Q, K, V for the current layer
                Q, K, V = load_qkv_data(
                    data_path, layer_idx, context_length, device, dtype, head_id=head_id
                )

                # Log tensor shapes and statistics
                Logger.log(f"Layer {layer_idx} - Tensor Shapes and Statistics:")
                Logger.log(f"  Q shape: {Q.shape}, Mean: {Q.abs().mean():.6f}")
                Logger.log(f"  K shape: {K.shape}, Mean: {K.abs().mean():.6f}")
                Logger.log(f"  V shape: {V.shape}, Mean: {V.abs().mean():.6f}")

                # Benchmark attention methods
                results, profile = benchmark_attention_methods(Q, K, V, layer_idx, context_length)

                for thresh in CONFIG["recall_thresholds"]:
                    recall_obj[context_length].setdefault(thresh, {})
                    record_recall_ratios(
                        results, profile, thresh, layer_idx, head_id, recall_obj[context_length][thresh]
                    )

    # recall_obj structure: {context_length: {threshold: {method: {layer_idx: {head_id: value}}}}}
    basedir = create_timestamped_directory(CONFIG["output_dir"])
    save_recall_and_sparsity_data_to_files(recall_obj, basedir)


def main():
    """Main function to initiate benchmarking across all layers."""
    benchmark_all_layers()


if __name__ == "__main__":
    main()
