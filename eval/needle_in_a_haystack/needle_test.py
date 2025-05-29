import argparse
import os
from dataclasses import dataclass
from datetime import datetime

from .needle_tools import LLMNeedleHaystackTester
from .needle_viz import plot_needle_viz

@dataclass
class Config:
    # Download example: wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl
    haystack_file: str = "./data/pg19_mini.jsonl"  # Path to the haystack file
    model_name: str = "Llama-3.1-8B-Instruct"
    run_name: str = None  # Name of the run, used for the output file
    context_lengths_min: int = 1_000
    context_lengths_max: int = 100_000
    n_context_length_intervals: int = 15  # Number of intervals between minimum and maximum context lengths
    n_document_depth_intervals: int = 10  # Number of intervals for needle position within the document
    n_rounds: int = 3
    seed: int = 42
    output_path: str = "results/needle/"
    jobs: str = None
    trust_remote_code: bool = False
    device: str = "cuda:0"
    pattern: str = "anchor_attn"
    pattern_config: str = None

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # Build output file name based on model, pattern, and pattern configuration
        file_suffix = ""
        if self.pattern_config:
            import json
            cfg = json.loads(self.pattern_config)  # Parse JSON string of pattern configuration
        output_file = f"{self.model_name.split('/')[-1]}_{self.pattern}.json"
        self.output_file = os.path.join(self.output_path, output_file)


def main(
    model_name: str,
    run_name: str = None,
    output_path: str = "results/needle/",
    rounds: int = 3,
    jobs: str = None,
    max_length: int = 100000,
    min_length: int = 1000,
    trust_remote_code: bool = False,
    device: str = "cuda:0",
    pattern: str = "anchor_attn",
    n_context_length_intervals: int = 15,
    n_document_depth_intervals: int = 10,
    pattern_config: str = None,
):
    # Initialize configuration
    config = Config(
        model_name=model_name,
        run_name=run_name,
        output_path=output_path,
        n_rounds=rounds,
        jobs=jobs,
        context_lengths_min=min_length,
        context_lengths_max=max_length,
        trust_remote_code=trust_remote_code,
        device=device,
        pattern=pattern,
        pattern_config=pattern_config,
        n_context_length_intervals=n_context_length_intervals,
        n_document_depth_intervals=n_document_depth_intervals,
    )
    
    # Create tester instance (only pass kwargs if using vLLM attention type)
    ht = LLMNeedleHaystackTester(config)
    print(os.path.abspath(__file__))
    print(f"config: {config}")
    # Start the needle-in-haystack tests
    ht.start_test()

    # Generate visualization plot
    print("making plot...")
    plot_needle_viz(
        config.output_file,
        (
            f"{config.model_name.split('/')[-1]}_{config.pattern}"
            if config.run_name is not None
            else ""
        ),
        config.context_lengths_min,
        config.context_lengths_max,
        output_path=config.output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="your_model_path/Llama-3.1-8B-Instruct",
        help="Path or name of the model to test"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for the run")
    parser.add_argument(
        "--n_context_length_intervals",
        type=int,
        default=15,
        help="Number of context length intervals"
    )
    parser.add_argument(
        "--n_document_depth_intervals",
        type=int,
        default=10,
        help="Number of document depth intervals"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/needle/",
        help="Directory to save results and plots"
    )
    parser.add_argument(
        "--pattern_path",
        type=str,
        default=None,
        help="Path to the attention pattern configuration JSON"
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of test rounds to run")
    parser.add_argument("--jobs", type=str, default=None, help="Optional job identifiers")
    parser.add_argument("--max_length", type=int, default=100000, help="Maximum context length to test")
    parser.add_argument("--min_length", type=int, default=1000, help="Minimum context length to test")
    parser.add_argument("--kv_cache_cpu", action="store_true", help="Enable KV cache on CPU")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow trusting remote code when loading the model"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device to use")
    parser.add_argument("--attn_type", type=str, default=None, help="Attention type (e.g., vllm)")
    parser.add_argument("--kv_type", type=str, default=None, help="KV cache type (e.g., dense)")
    parser.add_argument(
        "--pattern",
        type=str,
        default="anchor_attn",
        choices=[
            "default", "flash", "streaming_llm", "minfer", "vertical_slash", "flex_prefill", "anchor_attn", "anchor_attn_lite"
        ],
        help="The attention pattern to use in patch_model"
    )
    parser.add_argument(
        "--pattern_config",
        type=str,
        default=None,
        help="JSON string for additional pattern configuration"
    )
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        run_name=args.run_name,
        output_path=args.output_path,
        rounds=args.rounds,
        jobs=args.jobs,
        max_length=args.max_length,
        min_length=args.min_length,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        pattern=args.pattern,
        n_context_length_intervals=args.n_context_length_intervals,
        n_document_depth_intervals=args.n_document_depth_intervals,
        pattern_config=args.pattern_config,
    )
