import torch
import time
from .flash_attn_triton import triton_flash_attention
from .sparse_attn_with_pytorch import sparse_attn_with_torch, sparse_attn_topcdf_with_torch, sparse_attn_topk_with_torch
from my_utils.ops_util import load_data, get_deff
from my_utils import Logger
import os
import pandas as pd

# 统一的 config 字典，仅包含 sparse 相关注意力方法的参数配置
CONFIG = {
    "recall_thresholds": [0.01, 0.005],
    "output_dir": "./analysis_results",
    "sparse_topcdf_attention": {
        "cdf_thresholds": [0.95]  # Sparse TopCDF Attention 的 cdf_threshold 参数
        # "cdf_thresholds": []
    },
    "sparse_topk_attention": {
        "top_k_values": [4096,8192,16*1024]  # Sparse TopK Attention 的 top_K 参数
    },
    "sparse_attention": {
        "diff_anchors": [10,11] ,
    }
}

def time_attention_method(method, *args, num_runs=1, **kwargs):
    """
    Times the execution of an attention method over multiple runs.

    Args:
        method: The attention function to benchmark.
        num_runs: Number of runs to average over (default: 2).
        *args, **kwargs: Arguments to pass to the method.

    Returns:
        tuple: (output of the method, list of execution times in milliseconds, optional sparsity ratio)
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
        if isinstance(result, tuple):  # Handle methods returning (output, sparsity_ratio)
            output, sparsity_ratio = result
        else:
            output = result
    return output, times, sparsity_ratio

def print_timing_results(method_name, times, layer_idx, sparsity_ratio=None):
    """
    Prints timing results and optional sparsity ratio in a formatted manner for a specific layer.

    Args:
        method_name: Name of the attention method.
        times: List of execution times in milliseconds.
        layer_idx: Index of the layer being benchmarked.
        sparsity_ratio: Optional sparsity ratio to print.
    """
    avg_time = sum(times) / len(times)
    Logger.log(f"Layer {layer_idx} - {method_name}:")
    for i in range(len(times)):
        Logger.log(f"  Times: {times[i]:.2f} ms")
    Logger.log(f"  Average: {avg_time:.2f} ms")
    if sparsity_ratio is not None:
        Logger.log(f"  Sparsity Ratio: {sparsity_ratio:.6f}")

def print_diff_results(reference, method_output, method_name, layer_idx, results):
    """
    Prints the difference between reference and method output for a specific layer, including sparsity ratio.

    Args:
        reference: Reference output tensor.
        method_output: Output tensor from the method.
        method_name: Name of the attention method.
        layer_idx: Index of the layer being benchmarked.
        results: Dictionary containing the results, including sparsity ratios.
    """
    method_key = method_name.lower().replace(" ", "_")
    sparsity_ratio = results[method_key]["sparsity_ratio"] if method_key in results else None
    
    Logger.log(f"Layer {layer_idx} - {method_name} Difference:")
    get_deff(reference, method_output)
    if sparsity_ratio is not None:
        Logger.log(f"  Sparsity Ratio in Comparison: {sparsity_ratio:.6f}")

def load_qkv_data(data_path, layer_idx, context_length, device, dtype, head_id=-1,dataset='niah_single_1'):
    """
    Loads Q, K, V tensors for a specific layer.

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
        dataset=dataset
    )
    return Q, K, V

# 独立测试函数：Sparse TopCDF Attention
def test_sparse_topcdf_attention(Q, K, V, layer_idx, results):
    cdf_thresholds = CONFIG["sparse_topcdf_attention"]["cdf_thresholds"]
    
    for cdf in cdf_thresholds:
        method_name = f"Sparse TopCDF Attention (cdf_threshold={cdf})"
        
        method = lambda: sparse_attn_topcdf_with_torch(Q, K, V, cdf_threshold=cdf, fullkv=False, return_computational_ratio=True)
        Logger.log(f"调用 sparse_attn_topcdf_with_torch TopCDF={cdf}")
        output, times, sparsity_ratio = time_attention_method(method)
        
        # 处理输出
        if isinstance(output, tuple):
            output, sparsity_ratio = output
        
        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")
        
        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio
        }

# 独立测试函数：Sparse TopK Attention
def test_sparse_topk_attention(Q, K, V, layer_idx, results):
    top_k_values = CONFIG["sparse_topk_attention"]["top_k_values"]
    
    for k in top_k_values:
        method_name = f"Sparse TopK Attention (top_K={k})"
        
        method = lambda: sparse_attn_topk_with_torch(Q, K, V, top_K=k, fullkv=False, return_computational_ratio=True)
        Logger.log(f"调用 sparse_attn_topk_with_torch，top_K={k}")
        output, times, sparsity_ratio = time_attention_method(method)
        
        # 处理输出
        if isinstance(output, tuple):
            output, sparsity_ratio = output
        
        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")
        
        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio
        }

# 独立测试函数：Sparse Attention
def test_sparse_attention(Q, K, V, layer_idx, results):
    diff_anchors = CONFIG["sparse_attention"]["diff_anchors"]
    
    for diff in diff_anchors:
        method_name = f"Sparse Attention (diff_anchor={diff})"
        
        method = lambda: sparse_attn_with_torch(Q, K, V, diff_anchor=diff, fullkv=False, return_computational_ratio=True)
        Logger.log(f"调用 Sparse Attention diff_anchor={diff}")
        output, times, sparsity_ratio = time_attention_method(method)
        
        # 处理输出
        if isinstance(output, tuple):
            output, sparsity_ratio = output
        
        print_timing_results(method_name, times, layer_idx, sparsity_ratio)
        Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")
        
        results[method_name.lower().replace(" ", "_")] = {
            "output": output,
            "times": times,
            "sparsity_ratio": sparsity_ratio
        }

def benchmark_attention_methods(Q, K, V, layer_idx, context_length):
    """
    Benchmarks specified sparse attention methods for a given set of Q, K, V tensors and layer.

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

    # Flash Attention 作为基准
    flash_config = {
        "name": "Flash Attention",
        "method": lambda: triton_flash_attention(
            Q.transpose(1, 2).to(torch.bfloat16),
            K.transpose(1, 2).to(torch.bfloat16),
            V.transpose(1, 2).to(torch.bfloat16),
            causal=True
        ).transpose(1, 2),
        "sparsity": False
    }
    output, times, _ = time_attention_method(flash_config["method"])
    print_timing_results(flash_config["name"], times, layer_idx)
    Logger.log(f"  Output Mean: {output.abs().mean():.6f}\n")
    results["flash_attention"] = {
        "output": output,
        "times": times,
        "sparsity_ratio": None
    }
    profile.append(flash_config)

    # 测试 Sparse Attention
    Logger.log("\n=== Testing Sparse Attention ===")
    test_sparse_attention(Q, K, V, layer_idx, results)
    profile.extend([{"name": f"Sparse Attention (diff_anchor={d})", "sparsity": True} for d in CONFIG["sparse_attention"]["diff_anchors"]])

    # 测试 Sparse TopCDF Attention
    Logger.log("\n=== Testing Sparse TopCDF Attention ===")
    test_sparse_topcdf_attention(Q, K, V, layer_idx, results)
    profile.extend([{"name": f"Sparse TopCDF Attention (cdf_threshold={cdf})", "sparsity": True} for cdf in CONFIG["sparse_topcdf_attention"]["cdf_thresholds"]])

    # 测试 Sparse TopK Attention
    Logger.log("\n=== Testing Sparse TopK Attention ===")
    test_sparse_topk_attention(Q, K, V, layer_idx, results)
    profile.extend([{"name": f"Sparse TopK Attention (top_K={k})", "sparsity": True} for k in CONFIG["sparse_topk_attention"]["top_k_values"]])

    return results, profile

def get_final_ratio(sparsity_raw_value, method_key):
    final_sparsity_value = None  # 初始化

    if sparsity_raw_value is None:
        print(f"Warning: 'sparsity_ratio' not found for method {method_key}. Using NaN.")
        final_sparsity_value = float('nan')
    elif isinstance(sparsity_raw_value, torch.Tensor):
        if sparsity_raw_value.numel() == 1:  # 确保是单元素张量
            final_sparsity_value = sparsity_raw_value.item()
        else:
            print(f"Warning: 'sparsity_ratio' for method {method_key} is a multi-element tensor: {sparsity_raw_value}. Using NaN.")
            final_sparsity_value = float('nan')  # 或者其他处理方式，比如 .tolist() 或报错
    elif isinstance(sparsity_raw_value, (float, int)):  # 如果它已经是 float 或 int
        final_sparsity_value = float(sparsity_raw_value)  # 直接使用，确保是 float
    else:
        print(f"Warning: 'sparsity_ratio' for method {method_key} has unexpected type: {type(sparsity_raw_value)}. Using NaN.")
        final_sparsity_value = float('nan')
    return final_sparsity_value

# 返回对应门限下的recall 以及 稀疏率
def record_recall_ratios(results, profile, recall_threshold, layer_idx, head_idx, recall_obj):
    flash_key = "flash_attention"
    flash_output = results[flash_key]["output"]
    # Logger.log(f"flash_output:{flash_output.shape}")
    Logger.log(f"head_idx:{head_idx}")
    # Logger.log(f"flash_output:{flash_output[0,:,0,:3]}")
    if flash_output.size(1) != 1:
        flash_output = flash_output[:, head_idx:head_idx+1, :, :]
    # Logger.log(f"flash_output:{flash_output[0,:,0,:3]}")
    

    for config in profile:
        method_name = config["name"]
        method_key = method_name.lower().replace(" ", "_") 

        if method_key == flash_key:  # Skip Flash Attention itself
            continue
        if method_key in results:
            ref = results[method_key]["output"]
            # Logger.log(f"flash_output:{flash_output.shape}")
            if ref.size(1) != 1:
                ref = ref[:, head_idx:head_idx+1, :, :] 
            diff = torch.abs(ref - flash_output) 
            # Logger.log(f"ref:{ref.shape}")
            # Logger.log(f"flash_output:{flash_output.shape}")
            # Logger.log(f"ref:{ref[0,:,0,:3]}")
            if method_key not in recall_obj:
                recall_obj[method_key] = {}
            if layer_idx not in recall_obj[method_key]:
                recall_obj[method_key][layer_idx] = {}
            spa_ratio = results[method_key]["sparsity_ratio"]
            spa_ratio = get_final_ratio(spa_ratio, method_key)
            recall_obj[method_key][layer_idx][head_idx] = [((diff < recall_threshold).sum().item() / ref.numel()), spa_ratio]
            Logger.log(f"recall_obj[method_key][layer_idx][head_idx]:{recall_obj[method_key][layer_idx][head_idx]}")

def compare_attention_outputs(results, profile, layer_idx, head_idx):
    """
    Compares the outputs of different attention methods against Flash Attention for a specific layer.

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
            print_diff_results(flash_output, results[method_key]["output"], method_name, layer_idx, results)
        else:
            Logger.log(f"Layer {layer_idx} - {method_name} not found in results.")

def save_recall_and_sparsity_data_to_files(recall_obj, base_dir="recall_data_output"):
    """
    将 recall_obj 字典中的数据保存到指定目录结构的文件中。
    recall_obj 结构: {context_length: {threshold: {method: {layer_idx: {head_idx: [recall_value, sparsity_value]}}}}}

    输出目录结构:
    base_dir/
    ├── {context_length}/
    │   ├── {threshold}/
    │   │   ├── {method_name}_recall.csv  (存储第一个值)
    │   │   ├── {method_name}_sparsity.csv (存储第二个值)
    │   │   └── ...
    │   └── ...
    └── ...
    """
    print(f"Base directory for this run: {os.path.abspath(base_dir)}")

    for context_length, cl_data in recall_obj.items():
        cl_str = str(context_length)
        cl_dir_path = os.path.join(base_dir, cl_str)
        os.makedirs(cl_dir_path, exist_ok=True)
        print(f"  Created/Ensured directory: {cl_dir_path}")

        for threshold, th_data in cl_data.items():
            threshold_str = str(threshold).replace('.', '_')
            th_dir_path = os.path.join(cl_dir_path, threshold_str)
            os.makedirs(th_dir_path, exist_ok=True)
            print(f"    Created/Ensured directory: {th_dir_path}")

            for method_name, method_layer_data in th_data.items():
                safe_method_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in method_name)
                
                # --- 准备数据 ---
                all_layer_indices = sorted(list(method_layer_data.keys()))
                if not all_layer_indices:
                    print(f"      Skipping {method_name} for CtxLen={context_length}, Thresh={threshold}: No layer data.")
                    continue

                all_head_ids_set = set()
                has_valid_data_for_method = False
                for layer_idx in all_layer_indices:
                    if isinstance(method_layer_data[layer_idx], dict):
                        for head_id, values in method_layer_data[layer_idx].items():
                            if isinstance(values, (list, tuple)) and len(values) >= 1:  # 至少有一个值
                                all_head_ids_set.add(head_id)
                                has_valid_data_for_method = True
                    else:
                        print(f"      Warning: Expected dict for layer {layer_idx} in {method_name}, got {type(method_layer_data[layer_idx])}. Skipping this layer.")
                
                if not has_valid_data_for_method or not all_head_ids_set:
                    print(f"      Skipping {method_name} for CtxLen={context_length}, Thresh={threshold}: No valid head data found across layers.")
                    continue
                
                sorted_head_ids = sorted(list(all_head_ids_set), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else str(x))

                # 初始化两个数据列表，一个用于recall，一个用于sparsity
                recall_table_data = []
                sparsity_table_data = []
                
                found_recall_data = False
                found_sparsity_data = False

                for layer_idx in all_layer_indices:
                    recall_row_data = {'Layer_ID': f"Layer_{layer_idx}"}
                    sparsity_row_data = {'Layer_ID': f"Layer_{layer_idx}"}
                    
                    head_data_for_layer = method_layer_data.get(layer_idx, {})

                    for head_id in sorted_head_ids:
                        column_name = f"Head_{head_id}"
                        values = head_data_for_layer.get(head_id)

                        recall_value = float('nan')
                        sparsity_value = float('nan')

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

                # --- 保存 Recall CSV ---
                if found_recall_data and recall_table_data:
                    recall_csv_filename = f"{safe_method_name}_recall.csv"
                    recall_csv_filepath = os.path.join(th_dir_path, recall_csv_filename)
                    df_recall = pd.DataFrame(recall_table_data)
                    if 'Layer_ID' in df_recall.columns:
                        df_recall = df_recall.set_index('Layer_ID')
                    try:
                        df_recall.to_csv(recall_csv_filepath)
                        print(f"      Saved recall data to: {recall_csv_filepath}")
                    except Exception as e:
                        print(f"      Error saving recall data {recall_csv_filepath}: {e}")
                else:
                    print(f"      No recall data to write for {safe_method_name} (CtxLen={context_length}, Thresh={threshold})")

                # --- 保存 Sparsity CSV ---
                if found_sparsity_data and sparsity_table_data:
                    sparsity_csv_filename = f"{safe_method_name}_sparsity.csv"
                    sparsity_csv_filepath = os.path.join(th_dir_path, sparsity_csv_filename)
                    df_sparsity = pd.DataFrame(sparsity_table_data)
                    if 'Layer_ID' in df_sparsity.columns:
                        df_sparsity = df_sparsity.set_index('Layer_ID')
                    try:
                        df_sparsity.to_csv(sparsity_csv_filepath)
                        print(f"      Saved sparsity data to: {sparsity_csv_filepath}")
                    except Exception as e:
                        print(f"      Error saving sparsity data {sparsity_csv_filepath}: {e}")
                else:
                    print(f"      No sparsity data to write for {safe_method_name} (CtxLen={context_length}, Thresh={threshold})")

def create_timestamped_directory(parent_dir="."):
    """
    在指定的父目录下创建一个以当前时间命名的目录。
    目录名格式: YYYY-MM-DD_HH-MM-SS
    返回创建的目录的绝对路径。
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dir_path = os.path.join(parent_dir, timestamp)
    os.makedirs(new_dir_path, exist_ok=True)
    return os.path.abspath(new_dir_path)

def benchmark_all_layers():
    """
    Benchmarks sparse attention mechanisms for layers 0 to 31 and compares their outputs.
    """
    # Configuration
    Logger.set_log_file_path(f"profile_ops/{time.time()}.log")
    torch.manual_seed(42)
    B, H, N, D = 1, 32, 1024 * 4, 128
    global context_length
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    data_path = "./tensor"
    
    recall_obj = {}
    for c in [1, 2, 4, 8, 16, 32, 64, 128]:
        context_length = c * 1024
        if context_length not in recall_obj:
            recall_obj[context_length] = {}
        for layer_idx in range(32): 
            for head_id in range(32):
                # niah_single_1
                # Q, K, V = load_qkv_data(data_path, layer_idx, context_length, device, dtype, head_id=head_id,dataset="niah_single_1")
                Q, K, V = load_qkv_data(data_path, layer_idx, context_length, device, dtype, head_id=head_id,dataset="niah_multikey_3")
                
                # Benchmark attention methods
                results, profile = benchmark_attention_methods(Q, K, V, layer_idx, context_length)
                for threshold in CONFIG["recall_thresholds"]:
                    if threshold not in recall_obj[context_length]:
                        recall_obj[context_length][threshold] = {}
                    record_recall_ratios(results, profile, threshold, layer_idx, head_id, recall_obj[context_length][threshold])

    basedir = create_timestamped_directory(CONFIG["output_dir"])
    save_recall_and_sparsity_data_to_files(recall_obj, basedir)

def main():
    """Main function to initiate benchmarking across all layers."""
    benchmark_all_layers()

if __name__ == "__main__":
    main()