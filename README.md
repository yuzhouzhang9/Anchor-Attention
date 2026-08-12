

# AnchorAttn: Difference-Aware Sparse Attention with Stripe Granularity

AnchorAttn is a dynamic sparse attention implementation for accelerating the
prefill phase of long-context LLM inference. It identifies important attention
regions at stripe granularity, then computes sparse attention with Triton kernels
while preserving the model's HuggingFace generation interface.

The method is based on three stages:

1. **Pattern-based anchor computation**: compute anchor scores from initial and
   local attention regions, and cache online-softmax states.
2. **Difference-aware stripe sparsity identification**: compare compressed query
   scores against anchor scores to select important discrete KV coordinates.
3. **Fine-grained sparse computation**: compute final sparse attention over the
   selected coordinates while reusing the cached online-softmax states.

Compared with block-level sparse attention methods, AnchorAttn uses finer
stripe-level sparsity to reduce unnecessary computation in long-context prefill.

## Installation

```bash
conda create -n anchorattn python=3.10
conda activate anchorattn
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run AnchorAttn:

```bash
CUDA_VISIBLE_DEVICES=0 python test_hf.py \
  --model_path your_model_path/Llama-3.1-8B-Instruct \
  --pattern anchorattn \
  --config '{"theta":12,"step":16,"block_size_M":128}'
```

Run a baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python test_hf.py \
  --model_path your_model_path/Llama-3.1-8B-Instruct \
  --pattern baseline_flex_prefill \
  --config '{"block_size":128,"flex_prefill_gamma":0.95,"flex_prefill_tau":0.1}'
```

## Test Data

`test_hf.py` reads JSONL files from `data/test_data`.

Each sample uses:

```json
{"input": "question or prompt text", "output": "reference answer"}
```

Available test sets:

```text
niah_single_1
niah_multikey_3
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python test_hf.py \
  --model_path your_model_path/Llama-3.1-8B-Instruct \
  --pattern anchorattn \
  --data_name niah_single_1 \
  --output_path outputs/test_data_anchorattn.jsonl
```


## Supported Models

- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct

## Citation

If you use this work, please cite the ACL Anthology paper:

```bibtex
@inproceedings{zhang-etal-2025-anchorattention,
    title = "{A}nchor{A}ttention: Difference-Aware Sparse Attention with Stripe Granularity",
    author = "Zhang, Yu and
      Guo, Dong and
      Wu, Fang and
      Zhu, Guoliang and
      Ding, Dian and
      Zhang, Yiming",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.430/",
    doi = "10.18653/v1/2025.emnlp-main.430",
    pages = "8537--8549",
    ISBN = "979-8-89176-332-6"
}
```
