import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

# import kornia
from lavis.models import load_model_and_preprocess
sys.path.append("/home/xjg/VCD")
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    question_file = os.path.expanduser(args.question_file)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(question_file, "r") as f:
        lines = f.readlines()

    with open(answers_file, "w") as ans_file:
        for line in tqdm(lines):
            parts = line.strip().split("\t")
            image_file, qs, _ = parts[0], parts[1], parts[2]

            prompt = qs

            raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            # prepare the image
            image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            ## create a white image for contrastive decoding
            if args.use_cd:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            else:
                image_tensor_cd = None      

            with torch.inference_mode():
                outputs = model.generate({"image": image_tensor, "prompt": prompt},
                    use_nucleus_sampling=True, num_beams=1,
                    top_p=args.top_p, repetition_penalty=1,
                    images_cd=image_tensor_cd, cd_beta=args.cd_beta)

            outputs = outputs[0]

            new_line = line.strip() + "\t" + outputs + "\n"
            ans_file.write(new_line)
            ans_file.flush()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer_copy.txt")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=True)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
