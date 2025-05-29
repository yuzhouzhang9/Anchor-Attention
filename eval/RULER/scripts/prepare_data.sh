GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR=${1}
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=32  # increase to improve GPU utilization

# Model and Tokenizer
source config_models.sh
MODEL_NAME=${2}

MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=synthetic
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    SETTINGS_INFO=""
    if [[ -n ${METRIC} ]]; then SETTINGS_INFO+="${METRIC#--metric }_"; fi
    if [[ -n ${STRIDE} ]]; then SETTINGS_INFO+="fuse_${STRIDE##* }_"; fi
    if [[ -n ${THRESHOLD} && -z ${PRECISE_THRESHOLD} ]]; then SETTINGS_INFO+="thresh_${THRESHOLD#--threshold }_"; fi
    
    RESULTS_DIR="${ROOT_DIR}/${SETTINGS_INFO}${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
    done

    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
