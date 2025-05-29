import os
import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_utils.load_test_data import get_test_data
from anchor_attn import patch_model
import argparse

def main(
        device="cuda:0",
        pattern="flash"
):
    data, mx_len = get_test_data()
    answers = []
    preds = []
    for idx, item in enumerate(data):
        prompts = item["prompt"]
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
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
        pred = response
        print(f"context_length: {context_length}")
        print(f"pred: [{pred}]")
        print(f"answer: {item['outputs']}")
        answers.append(item['outputs'])
        preds.append(pred)

        # Add to results list for JSON output
        result = {
            "pred": pred,
            "answer": item['outputs'],
            "pattern": pattern,
            "idx": idx,
            "context_length": context_length,
            "prefill_time": prefill_time,
            "tot_time": tot_time,
            "config":args.config
        }
        try:
            with open("data/result.jsonl", "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")  # JSONL 每 row 一个对象
        except Exception as e:
            print(f"Failed to write to result.jsonl: {e}")

    for i in range(len(answers)):
        print(f"pred: [{preds[i]}], answer: {answers[i]}")

    print("Results appended to data/result.jsonl")

if __name__ == "__main__":
    from my_utils import Logger
    import time
    Logger.set_log_file_path(f"log/test_hf/{time.time()}.log")
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
        default="anchor_attn",  # 默认模式
        choices=[
            "default", "flash", "streaming_llm", "minfer", "vertical_slash", "flex_prefill", "anchor_attn", "anchor_attn_lite", 
        ],
        help="The attention pattern to use in patch_model. Default is 'anchor_attn'."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None
    )
    # python test_hf.py --pattern flex_prefill --config '{"block_size": 128,"flex_prefill_gamma": 0.95,"flex_prefill_tau": 0.1,"flex_prefill_min_budget": 1024,"flex_prefill_max_budget": null}'
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
            pattern=args.pattern
        )
    except Exception as e:
        print(f"An error occurred: {e}")