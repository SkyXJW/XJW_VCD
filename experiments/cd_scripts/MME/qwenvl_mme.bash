export CUDA_VISIBLE_DEVICES=7

seed=${1:-75}
dataset_name=${2:-"text_translation"}
model_path=${4:-"/home/xjg/checkpoints/Qwen_VL"}
cd_alpha=${4:-1}
cd_beta=${5:-0.2}
noise_step=${6:-500}

python ./eval/MME/object_hallucination_vqa_qwenvl.py \
--model-path ${model_path} \
--question-file ./eval/MME/tools/eval_tool/Your_Results/${dataset_name}.txt \
--image-folder ./data/MME/${dataset_name} \
--answers-file ./output/answer_files_MME/qwenvl/cd/seed${seed}/${dataset_name}.txt \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}


