
export PYTHONPATH=$PYTHONPATH:$(pwd) 
export CUDA_VISIBLE_DEVICES=0
cd eval/RULER
# bash setup.sh
cd scripts
model_path=${1-"your_model_path/model"}
model_name=${2-"Llama-3.1-8B-Instruct"}
bash prepare_data.sh $model_path $model_name # ~/model # Llama-3.1-8B-Instruct

model_path=${1-"your_model_path/model"}
model_name=${2-"Llama-3.1-8B-Instruct"}
pattern=${3-"anchor_attn"}
config=${4-'{"theta":12,"step":16}'}

bash pred.sh $model_path $model_name $pattern $config # ~/model 、Llama-3.1-8B-Instruct 、 pattern
bash eval.sh $model_name $pattern

#