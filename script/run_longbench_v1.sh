# 
mkdir -p ./data/LongBench
cd data/LongBench
wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip data.zip
cd ../..
# longbench data needs to be downloaded
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
TASKS=(
    "narrativeqa"
    "lcc"
    "multifieldqa_en"
    "multifieldqa_zh"
    "hotpotqa"
    "2wikimqa"
    "musique"
    "dureader"
    "gov_report"
    "qmsum"
    "multi_news"
    "vcsum"
    "trec"
    "triviaqa"
    "samsum"
    "lsht"
    "passage_count"
    "passage_retrieval_en"
    "passage_retrieval_zh"
    "repobench-p"
    "qasper"
)

# your model path
DEVICE="cuda:0"
MODEL_PATH="your_model_path/Llama-3.1-8B-Instruct"
pattern=anchor_attn
config='{"theta":12,"step":16}'
# pattern=flex_prefill
# pattern=vertical_slash

python -m eval.LongBench.pred_longbench_v1 \
    --device "$DEVICE" \
    --datasets "${TASKS[@]}" \
    --model_path "$MODEL_PATH" \
    --pattern "$pattern"\
    --config "$config"

echo "All evaluations completed!"

python -m eval.LongBench.eval --results_path pred_longbench_v1