export CUDA_VISIBLE_DEVICES=7

seed=${1:-50}
dataset_name=${2:-"coco"}
type=${3:-"random"}
model_path=${4:-"/home/xjg/checkpoints/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
#   image_folder=./data/coco/val2014
  image_folder=./data/POPE/coco/val2014
else
#   image_folder=./data/gqa/images
  image_folder=./data/POPE/gqa/images
fi

python ./experiments/eval/POPE/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file ./experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/answer_files_POPE/llava15_${dataset_name}_pope_${type}_answers_cd_seed${seed}.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}
# --use_cd \


