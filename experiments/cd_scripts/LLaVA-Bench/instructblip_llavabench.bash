export CUDA_VISIBLE_DEVICES=0

seed=${1:-55}
model_path=${3:-"/home/xjg/checkpoints/instructblip-vicuna-7b"}
cd_alpha=${4:-1}
cd_beta=${5:-0.2}
noise_step=${6:-500}

python ./eval/LLaVA-Bench/object_hallucination_vqa_instructblip.py \
--question-file ./data/LLaVA-Bench/questions.jsonl \
--image-folder ./data/LLaVA-Bench/images \
--answers-file ./output/answer_files_LLaVA-Bench/instructblip/answers_instructblip_cd.jsonl \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}