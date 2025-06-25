import os
import torch
from transformers import AutoModelForSeq2SeqLM

# 模型文件路径
model_path = os.path.abspath("/home/xjg/checkpoints/instructblip-vicuna-7b/")

# 来加载模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

# 保存模型的状态字典到 .pth 文件
torch.save(model.state_dict(), "instruct_blip_vicuna7b_trimmed.pth")

