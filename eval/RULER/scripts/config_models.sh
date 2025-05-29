TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    4096
    8192
    16384
    32768
    65536
    131072
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3

    case $MODEL_NAME in
        Llama-3.1-8B-Instruct)
            MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
    esac

    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
