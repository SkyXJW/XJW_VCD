import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gt_files", type=str, default="data/POPE/gqa/gqa_pope_random.json")
parser.add_argument("--gen_files", type=str, default="output/answer_files_POPE/qwenvl_gqa_pope_random_answers_cd_seed55.jsonl")
args = parser.parse_args()

# open ground truth answers
with open(os.path.expanduser(args.gt_files), "r") as f:
    gt_files = [json.loads(line) for line in f]



# open generated answers
gen_files = []
with open(os.path.expanduser(args.gen_files), "r") as f:
    for line in f:
        line = line.strip()
        if line:  # 确保行不为空
            try:
                gen_files.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line: {line}")


# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0

# compare answers
for index, line in enumerate(gt_files):
    idx = line["question_id"]
    gt_answer = line["label"]
    assert idx == gen_files[index]["question_id"]
    gen_answer = gen_files[index]["text"]
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions
# report results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'Accuracy: {accuracy}')
print(f'yes: {yes_proportion}')
print(f'unknow: {unknown_prop}')