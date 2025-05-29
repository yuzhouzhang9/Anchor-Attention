ROOT_DIR="benchmark_root" # Path storing generated task samples and model predictions.

MODEL_DIR=${1}
MODEL_NAME=${2-"Llama-3.1-8B-Instruct"}
ATTN_PATTERN=${3-"anchor_attn"} # Default attention pattern.
config=${4-'{"theta":12,"step":16}'}
# Model and Tokenizer
echo $config
source config_models.sh

MODEL_CONFIG=$(MODEL_SELECT "${MODEL_NAME}" "${MODEL_DIR}" ".")
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

# Benchmark and Tasks
source config_tasks.sh
BENCHMARK="synthetic" # Default to synthetic if not provided.
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Sequence Lengths
SEQ_LENGTHS=(
    4096
    8192
    16384
    32768
    65536
    131072
)
device=cuda:0
# Start Prediction
total_time=0
echo "path ${MODEL_PATH} "
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    # Run the Python script to generate predictions
    echo "Generating predictions for ${MODEL_NAME} on ${BENCHMARK} with sequence length ${MAX_SEQ_LENGTH}..."
    
    start_time=$(date +%s)
    python pred.py \
        --device "${device}" \
        --datasets "${TASKS[@]}" \
        --context "${MAX_SEQ_LENGTH}" \
        --model_path "${MODEL_PATH}" \
        --pattern "${ATTN_PATTERN}" \
        --ROOT_DIR "${ROOT_DIR}" \
        --BENCHMARK "${BENCHMARK}"\
        --config "${config}"
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    total_time=$((total_time + elapsed_time))
    echo "Completed predictions for sequence length ${MAX_SEQ_LENGTH}. Time taken: ${elapsed_time} seconds."
done

echo "Total time spent on predictions: ${total_time} seconds"