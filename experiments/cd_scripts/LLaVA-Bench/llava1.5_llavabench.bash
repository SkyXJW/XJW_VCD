export CUDA_VISIBLE_DEVICES=1

seed=${1:-50}
model_path=${3:-"/home/xjg/checkpoints/llava-v1.5-7b"}
cd_alpha=${4:-1}
cd_beta=${5:-0.2}
noise_step=${6:-500}

python ./eval/LLaVA-Bench/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file ./data/LLaVA-Bench/questions.jsonl \
--image-folder ./data/LLaVA-Bench/images \
--answers-file ./output/answer_files_LLaVA-Bench/llava15/answers_llava15_cd.jsonl \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}