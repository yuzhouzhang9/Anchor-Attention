import os
import time
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_utils import Logger, get_longbenchv1
from anchor_attn import patch_model
import argparse

def main(
    device: str = "cuda:0",
    datasets: list[str] = ["fwe"],
    model_name: str = "llama3",
    pattern: str = "anchor_attn"
):
    """
    Iterate through specified datasets, generate model responses for each prompt,
    and append results to per-dataset JSONL files, skipping already processed items.

    Args:
        device (str): Torch device identifier (e.g., 'cuda:0', 'cpu').
        datasets (list[str]): List of dataset names for processing.
        model_name (str): Name of the model for output directory organization.
        pattern (str): Attention pattern identifier used when patching the model.
    """
    Logger.log(f"Datasets to process: {datasets}")
    for dataset in datasets:
        Logger.log(f"Processing dataset: {dataset}")
        # Load prompts and maximum generation length for this dataset
        data, max_length = get_longbenchv1(dataset)
        if not data:
            continue

        # Build output directory relative to script location
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_dir = os.path.join(base_name, model_name, pattern)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{dataset}.jsonl")
        processed_count = 0

        # If output exists, count already processed lines to skip duplicates
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        json.loads(line)
                        processed_count += 1
                    except json.JSONDecodeError:
                        continue
        Logger.log(f"Skipping first {processed_count} processed samples.")

        # Append new inferences to the JSONL file
        with open(output_file, "a", encoding="utf-8") as outfile:
            for item in tqdm(data[processed_count:], desc=f"Generating for {dataset}"):
                prompt_text = item["prompt"]
                # Tokenize the prompt
                tokenized = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
                prompt_len = tokenized.input_ids.shape[1]

                start = time.time()
                output_ids = model.generate(
                    **tokenized,
                    max_new_tokens=max_length,
                    num_beams=1,
                    do_sample=False
                )
                end = time.time()

                # Record timing and lengths
                item["generation_time"] = end - start
                response = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
                item["prompt_length"] = prompt_len
                item["response_length"] = output_ids.shape[1]
                item["pred"] = response
                # Truncate stored prompt for brevity
                item["prompt"] = prompt_text[:100] + "..."

                # Write JSON line
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                outfile.flush()

        Logger.log(f"Completed appending results to {output_file}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate model responses for LongBench v1 datasets.")
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device (e.g., 'cuda:0' or 'cpu')."
    )
    parser.add_argument(
        "--datasets", type=str, nargs='+', default=["lcc"],
        help="List of datasets to process."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="HuggingFace model path or identifier."
    )
    parser.add_argument(
        "--pattern", type=str, choices=[
            "default", "flash", "streaming_llm", "minfer", "vertical_slash",
            "flex_prefill", "anchor_attn", "anchor_attn_lite"
        ], default="anchor_attn",
        help="Attention pattern to apply."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional configuration parameters for patch_model."
    )
    args = parser.parse_args()

    # Initialize logger file dynamically under log/{script_name}/{timestamp}.log
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = int(time.time())
    Logger.set_log_file_path(f"{script_name}/{timestamp}.log")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left"
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
        use_cache=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()

    Logger.log(f"Arguments: {args}")
    # Patch the model's attention implementation
    patch_model(model=model, pattern=args.pattern, config=args.config)

    # Extract model name and run main loop
    model_name = os.path.basename(args.model_path)
    main(
        device=args.device,
        datasets=args.datasets,
        model_name=model_name,
        pattern=args.pattern
    )