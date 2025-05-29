import os
import time
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from anchor_attn import patch_model
import argparse

def get_ruler_data(ROOT_DIR, BENCHMARK, c_length, dataset, model_name):
    # Construct the path to the validation file for the given dataset and context length
    data_path = os.path.join(f"./{ROOT_DIR}/{model_name}/{BENCHMARK}/{c_length}/data/{dataset}", f"validation.jsonl")
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} does not exist.")
        return []
    else:
        print(f"File {data_path} exists.")
    # Read JSONL data
    with open(data_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def get_pred_path(ROOT_DIR, BENCHMARK, c_length, model_name, pred, pattern):
    # Construct the directory path for storing predictions
    return os.path.join(f"./{ROOT_DIR}/{model_name}/{BENCHMARK}/{c_length}/{pred}/{pattern}")
    
def main(
        device="cuda:0",
        datasets=["fwe"],
        context_length=[200],
        model_name="llama3",
        pattern="anchor_attn",
        ROOT_DIR="benchmark_root",
        BENCHMARK="synthetic",
        pred: str = "pred",
):
    for c_length in context_length:
        for dataset in datasets:
            # Load validation data for the current dataset
            data = get_ruler_data(ROOT_DIR, BENCHMARK, c_length, dataset, model_name)
            if not data:
                continue
            # Prepare output directory
            output_dir = get_pred_path(ROOT_DIR, BENCHMARK, c_length, model_name, pred, pattern)
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine output file name based on optional config
            output_file = os.path.join(output_dir, f"{dataset}.jsonl")
                
            # Count existing lines to avoid reprocessing
            cnt = 0
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    for line in f:
                        try:
                            json.loads(line.strip())
                            cnt += 1
                        except json.JSONDecodeError:
                            continue
            # Skip if all data already processed
            if cnt == len(data):
                continue
            print(f"Appending to output_file: {output_file}")
            if not os.path.exists(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Generate predictions for remaining items
            with open(output_file, "a") as f:
                for item in tqdm(data[cnt:], desc=f"Processing prompts for {dataset}"):
                    prompts = item["input"]
                    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    context_tokens = tokenized_prompts.input_ids.shape[1]
                    start_time = time.time()
                    # Generate response using the model
                    output = model.generate(
                        **tokenized_prompts,
                        max_new_tokens=128,
                        num_beams=1,
                        do_sample=False,
                    )
                    end_time = time.time()
                    item["tot_time"] = end_time - start_time
                    # Decode generated tokens
                    response = tokenizer.decode(output[0, context_tokens:], skip_special_tokens=True)
                    item["prompt_length"] = context_tokens
                    item["total_length"] = output.shape[1]
                    item["pred"] = response
                    # Truncate input for storage
                    item["input"] = item["input"][:100] + "..."
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
            print(f"Finished appending deduplicated results to {output_file}")

if __name__ == "__main__":
    from my_utils import Logger
    # Initialize logger with timestamped file path
    Logger.set_log_file_path(f"log/pred_ruler_test/{time.time()}.log")
    print("----------------------------------------")
    parser = argparse.ArgumentParser(description="Process prompts and generate responses.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g., 'cuda:0' or 'cpu'."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=[
            "niah_single_1", "niah_single_2", "niah_single_3",
            "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multivalue", "niah_multiquery", "vt", "cwe",
            "fwe", "qa_1", "qa_2"
        ],
        help="List of datasets to evaluate."
    )
    parser.add_argument(
        "--context",
        type=int,
        nargs='+',
        default=[1024 * i for i in [1, 2, 4, 8, 16, 32, 64, 128]],
        help="List of context lengths to test."
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="pred",
        help="Subdirectory name for predictions."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="your_model_path/Llama-3.1-8B-Instruct",
        help="Path to the pretrained model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON string of pattern-specific configurations."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="anchor_attn",
        choices=[
            "default", "flash", "streaming_llm", "minfer", "vertical_slash",
            "flex_prefill", "anchor_attn", "anchor_attn_lite"
        ],
        help="Attention pattern to apply with patch_model."
    )
    parser.add_argument(
        "--ROOT_DIR",
        type=str,
        default="benchmark_root",
        help="Root directory for data and predictions."
    )
    parser.add_argument(
        "--BENCHMARK",
        type=str,
        default="synthetic",
        help="Name of the benchmark to run."
    )

    args = parser.parse_args()
    print("----------------------------------------")
    print(f"Arguments: {args}")
    model_path = args.model_path
    # Load tokenizer and model with specified settings
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"model_path:{model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    # Apply attention pattern patch
    patch_model(model=model,
                pattern=args.pattern,
                config=args.config)
    model_name = os.path.basename(model_path)
    # Execute main processing function
    main(
        device=args.device,
        datasets=args.datasets,
        context_length=args.context,
        model_name=model_name,
        pattern=args.pattern,
        ROOT_DIR=args.ROOT_DIR,
        BENCHMARK=args.BENCHMARK,
    )
