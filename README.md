# AnchorAttention: Difference-Aware Sparse Attention with Stripe Granularity


## Abstract


Large Language Models (LLMs) with extended context lengths face significant computational challenges during the pre-filling phase, primarily due to the quadratic complexity of self-attention. Existing methods typically employ dynamic pattern matching and block-sparse low-level implementations. However, their reliance on local information for pattern identification fails to capture global contexts, and the coarse granularity of blocks leads to persistent internal sparsity, resulting in suboptimal accuracy and efficiency. To address these limitations, we propose \textbf{AnchorAttention}, a difference-aware, dynamic sparse attention mechanism that efficiently identifies critical attention regions at a finer stripe granularity while adapting to global contextual information, achieving superior speed and accuracy. AnchorAttention comprises three key components: (1) \textbf{Pattern-based Anchor Computation}, leveraging the commonalities present across all inputs to rapidly compute a set of near-maximum scores as the anchor; (2) \textbf{Difference-aware Stripe Sparsity Identification}, performing difference-aware comparisons with the anchor to quickly obtain discrete coordinates of significant regions in a stripe-like sparsity pattern; (3) \textbf{Fine-grained Sparse Computation}, replacing the traditional contiguous KV block loading approach with simultaneous discrete KV position loading to maximize sparsity rates while preserving full hardware computational potential.
With its finer-grained sparsity strategy, \textbf{AnchorAttention} achieves higher sparsity rates at the same recall level, significantly reducing computation time. Compared to previous state-of-the-art methods, at a text length of 128k, it achieves a speedup of 1.44$\times$ while maintaining higher recall rates. 



## Start

To install the required packages:

```bash
conda create -n anchorattn python=3.10
conda activate anchorattn
pip install -r requirements.txt
```


quick start

```bash
export CUDA_VISIBLE_DEVICES=0
python test_hf.py --model_path your_model_path --pattern anchor_attn
```

benchmark test

```bash
# Change your model path
bash script/run_longbench_v1.sh
bash script/run_needle.sh
bash script/run_ruler.sh
```


## Support model

Llama-3.1-8B-Instruct

Qwen2.5-7B-Instruct

## implementation

based on flexprefill and minference