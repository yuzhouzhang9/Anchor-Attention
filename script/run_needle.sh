export TOKENIZERS_PARALLELISM=false

# Load Haystack
mkdir -p data
wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl

model_name=${1-'your_model_path/Llama-3.1-8B-Instruct'}
pattern=${2-'anchor_attn_lite'}

python -m eval.needle_in_a_haystack.needle_test \
    --model_name $model_name \
    --max_length 128000\
    --min_length 16000 \
    --rounds 10 \
    --pattern $pattern\
    --output_path ./needle \
    --run_name $pattern \
    --jobs 0-15 \
    --device cuda:0\
    --n_document_depth_intervals 9\
    --n_context_length_intervals 9

#  Visualization for this diff_anchor value
mkdir -p figures
python -m eval.needle_in_a_haystack.needle_viz \
    --res_file "./needle/Llama-3.1-8B-Instruct_$pattern.json" \
    --model_name Llama-3.1-8B-Instruct \
    --pattern $pattern \
    --max_context 128000 \
    --min_context 16000