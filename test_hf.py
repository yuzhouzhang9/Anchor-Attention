import os
import sys
import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from patch import patch_model


def load_data(data_name="niah_multikey_3"):
    data_path = os.path.join("data/test_data", f"{data_name}.jsonl")
    res = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            res.append({
                "input": data["input"],
                "output": data["output"],
            })
    return res, 32


def main(
        device="cuda:0",
        pattern="anchorattn",
        data_name="niah_multikey_3",
        output_path="data/result.jsonl",
):
    data, mx_len = load_data(data_name)
    answers = []
    preds = []
    correct = 0
    for idx, item in enumerate(data):
        prompt = item["input"]
        target = item["output"]
        tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        context_length = tokenized_prompts.input_ids.shape[1]

        # Prefill phase: Measure time for processing the input prompt
        model.eval()  # Ensure model is in evaluation mode
        start_prefill_time = time.time()
        with torch.no_grad():
            # Simulate prefill by running a forward pass on the input prompt
            outputs = model.generate(
                **tokenized_prompts,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
            )
            # The forward pass computes the initial KV cache
        end_prefill_time = time.time()
        prefill_time = (end_prefill_time - start_prefill_time) * 1000  # Convert to milliseconds
        print(f"Prefill time: {prefill_time:.2f}ms")

        # Generation phase: Measure time for token generation
        start_generate_time = time.time()
        output = model.generate(
            **tokenized_prompts,
            max_new_tokens=mx_len,
            num_beams=1,
            do_sample=False,
        )
        end_generate_time = time.time()
        tot_time = (end_generate_time - start_generate_time) * 1000  # Convert to milliseconds
        print(f"Total time: {tot_time:.2f}ms")

        response = tokenizer.decode(output[0, context_length:], skip_special_tokens=True)
        prediction = response
        print(f"context_length: {context_length}")
        print(f"prediction: [{prediction}]")
        print(f"output: {target}")
        answers.append(target)
        preds.append(prediction)
        is_correct = target in prediction
        correct += int(is_correct)

        # Add to results list for JSON output
        result = {
            "input": prompt[:100] + "...",
            "output": target,
            "prediction": prediction,
            "data_name": data_name,
            "pattern": pattern,
            "idx": idx,
            "context_length": context_length,
            "prefill_time": prefill_time,
            "tot_time": tot_time,
            "config": args.config,
            "is_correct": is_correct,
        }
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")  # JSONL 每 row 一个对象
        except Exception as e:
            print(f"Failed to write to {output_path}: {e}")

    for i in range(len(answers)):
        print(f"prediction: [{preds[i]}], output: {answers[i]}")

    print(f"Accuracy: {correct}/{len(data)}")
    print(f"Results appended to {output_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process prompts and generate responses.")
    # Device and dataset arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to use (e.g., 'cuda:0', 'cuda:1', 'cpu'). Default is 'cuda:0'."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="your_model_path/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="anchorattn",  # 默认模式
        choices=[
            "baseline", "baseline_flash", "baseline_streaming_llm",
            "baseline_minference", "baseline_vertical_slash",
            "baseline_flex_prefill", "anchorattn",
        ],
        help="Attention method: anchorattn or one of the baseline_* methods."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="niah_multikey_3",
        choices=["niah_single_1", "niah_multikey_3"],
        help="Dataset name under data/test_data without the .jsonl suffix.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/result.jsonl",
        help="Path to append JSONL results.",
    )
    # python test_hf.py --pattern baseline_flex_prefill --config '{"block_size": 128,"flex_prefill_gamma": 0.95,"flex_prefill_tau": 0.1,"flex_prefill_min_budget": 1024,"flex_prefill_max_budget": null}'
    args = parser.parse_args()
    file_name = os.path.basename(__file__)
    model_path = args.model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a new padding token
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=args.device,  # Use the device specified in the command-line argument
            use_cache=True,
            attn_implementation="flash_attention_2",  # 需要使用flash_attention_2
        )
        model.eval()
        patch_model(model=model, pattern=args.pattern, config=args.config)
        model_name = model_path.split("/")[-1]
        main(
            device=args.device,
            pattern=args.pattern,
            data_name=args.data_name,
            output_path=args.output_path,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
