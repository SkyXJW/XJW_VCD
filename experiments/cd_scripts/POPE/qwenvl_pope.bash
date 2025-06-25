export CUDA_VISIBLE_DEVICES=1

seed=${1:-75}
dataset_name=${2:-"gqa"}
type=${3:-"adversarial"}
model_path=${4:-"/home/xjg/checkpoints/Qwen_VL"}
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

python ./eval/POPE/object_hallucination_vqa_qwenvl.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/answer_files_POPE/qwenvl_${dataset_name}_pope_${type}_answers_cd_seed${seed}.jsonl \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}


