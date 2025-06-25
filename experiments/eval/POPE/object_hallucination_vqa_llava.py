import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

# import kornia
from transformers import set_seed
sys.path.append("/home/xjg/VCD")
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
# evolve_vcd_sampling()

import pandas as pd
import yaml

def read_csv(file_name):
    try:
        df = pd.read_csv(file_name)

        # 返回读取的DataFrame
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")

    # df = read_csv("/home/xjg/TruthX/data/TruthfulQA.csv")
    # data = list(df.T.to_dict().values())
    # fold1_data = load_yaml("/home/xjg/TruthX/data/truthfulqa_data_fold1.yaml")["train_set"]

    with open("/home/xjg/myTruthX/data/dinm/SafeEdit/SafeEdit_test.json", 'r') as file:
        data = json.load(file)

    # 这里的64代表该模型由64个ATT与FFN模块组成
    common_representation_pos = [[] for _ in range(64)]
    common_representation_neg = [[] for _ in range(64)]
    i = 0
    # for line in tqdm(questions):
    for line in tqdm(data):
        # idx = line["question_id"]
        # image_file = line["image"]
        # qs = line["text"]

        # qs = line['Question']
        # correct_answer = line['Best Answer']
        # incorrect_answer = line['Incorrect Answers']
        
        i += 1
        if(i==51):
            break
        line = data[i-1]

        qs = line['question']
        correct_answer = line['safe generation']
        incorrect_answer = line['unsafe generation']
        
        # if i not in fold1_data:
        #     i += 1
        #     continues

        # cur_prompt = qs
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_ids1 = tokenizer_image_token(line['question'] + " " + correct_answer, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_ids2 = tokenizer_image_token(line['question'] + " " + incorrect_answer, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # image = Image.open(os.path.join(args.image_folder, image_file))
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # if args.use_cd:
        #     image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        # else:
        #     image_tensor_cd = None      

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.no_grad():
            correct_output = model(input_ids1,output_hidden_states=True,output_attentions=True)
            correct_att_hidden_states = correct_output.att_hidden_states # 多个[batch_size, num_tokens, representation_dim]
            correct_ffn_hidden_states = correct_output.ffn_hidden_states # 多个[batch_size, num_tokens, representation_dim]

            # incorrect_output = model(input_ids2,output_hidden_states=True,output_attentions=True)
            # incorrect_att_hidden_states = incorrect_output.att_hidden_states
            # incorrect_ffn_hidden_states = incorrect_output.ffn_hidden_states
        
        # 转换 input_ids 为 token
        correct_tokens = tokenizer.convert_ids_to_tokens(input_ids1.squeeze().tolist())
        incorrect_tokens = tokenizer.convert_ids_to_tokens(input_ids2.squeeze().tolist())

        # 将问题转换为token
        question_tokens = tokenizer.convert_ids_to_tokens(tokenizer_image_token(line['question'], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).squeeze().tolist())
        question_tokens_len = len(question_tokens)

        # 将回答转换为token
        correct_answer_tokens = tokenizer.tokenize(line['safe generation'])
        incorrect_answer_tokens = tokenizer.tokenize(line['unsafe generation'])

        # 找出同时出现在正确和错误回答中的 token
        common_tokens = set(correct_answer_tokens).intersection(set(incorrect_answer_tokens))

        for token in common_tokens:
            # 在正确和错误回答的 token 列表中找到相同 token 的索引
            correct_index = next((i for i, t in enumerate(correct_answer_tokens) if t == token), None)
            incorrect_index = next((i for i, t in enumerate(incorrect_answer_tokens) if t == token), None)

            # 提取正确回答的internal representation
            for layer in range(len(correct_att_hidden_states)):
                common_representation_pos[2*layer].append(correct_att_hidden_states[layer].squeeze(0)[correct_index+question_tokens_len])
                common_representation_pos[2*layer+1].append(correct_ffn_hidden_states[layer].squeeze(0)[correct_index+question_tokens_len])
            # # 提取错误回答的internal representation
            # for layer in range(len(incorrect_att_hidden_states)):
            #     common_representation_neg[2*layer].append(incorrect_att_hidden_states[layer].squeeze(0)[incorrect_index+question_tokens_len])
            #     common_representation_neg[2*layer+1].append(incorrect_ffn_hidden_states[layer].squeeze(0)[incorrect_index+question_tokens_len])
    
    # 将张量转换为 torch.float32
    common_representation_pos = [torch.stack(item) for item in common_representation_pos]
    common_representation_pos = torch.stack(common_representation_pos)
    common_representation_pos = common_representation_pos.to(torch.float32)
    torch.save(common_representation_pos, "/home/xjg/myTruthX/data/dinm/SafeEdit/llava/test_common_representations_pos.pth")
    
    # # 将张量转换为 torch.float32
    # common_representation_neg = [torch.stack(item) for item in common_representation_neg]
    # common_representation_neg = torch.stack(common_representation_neg)
    # common_representation_neg = common_representation_neg.to(torch.float32)
    # torch.save(common_representation_neg, "/home/xjg/myTruthX/data/dinm/SafeEdit/llava/test_common_representations_neg.pth")


    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids,
    #             images=image_tensor.unsqueeze(0).half().cuda(),
    #             images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
    #             cd_alpha = args.cd_alpha,
    #             cd_beta = args.cd_beta,
    #             do_sample=True,
    #             temperature=args.temperature,
    #             top_p=args.top_p,
    #             top_k=args.top_k,
    #             max_new_tokens=1024,
    #             use_cache=True)

    #     input_token_len = input_ids.shape[1]
    #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    #     outputs = outputs.strip()
    #     if outputs.endswith(stop_str):
    #         outputs = outputs[:-len(stop_str)]
    #     outputs = outputs.strip()

    #     ans_file.write(json.dumps({"question_id": idx,
    #                                "prompt": cur_prompt,
    #                                "text": outputs,
    #                                "model_id": model_name,
    #                                "image": image_file,
    #                                "metadata": {}}) + "\n")
    #     ans_file.flush()
    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)