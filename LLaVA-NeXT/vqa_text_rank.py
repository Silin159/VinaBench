
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import copy
import torch
import json
import numpy as np
import os
from torch.nn import functional as F
from tqdm import tqdm
import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov-sft"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
torch_dtype="float16"
img_dtype = torch.float16
tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, torch_dtype=torch_dtype)
llava_model.eval()

def get_rank(score, candidates):
    rank = 0
    while rank < len(candidates):
        if score < candidates[rank]:
            rank += 1
        else:
            break
    return rank

# take our experiments on MM-Interleaved as examples
root = "../MM-Interleaved/OUTPUT/$_inf/*/"
experiments = {"vwp": ["no_cons_ds250_full_inf_32k/gold",
                       "no_cons_ds250_full_inf_32k/pred",
                       "llama_cons_ds250_full_inf_30k/pred",
                       "gold_cons_ds250_full_inf_30k/pred"],
               "sb20k": ["no_cons_ds250_full_inf_32k/gold",
                         "no_cons_ds250_full_inf_32k/pred",
                         "llama_cons_ds250_full_inf_30k/pred",
                         "gold_cons_ds250_full_inf_30k/pred"],
               "salon_short": ["no_cons_ds250_full_inf_32k/gold",
                               "no_cons_ds250_full_inf_32k/pred",
                               "llama_cons_ds250_full_inf_36k/pred",
                               "gold_cons_ds250_full_inf_36k/pred"]}

model_ranks = {}
conv_template = "qwen_2"
for dataset in ["vwp", "sb20k", "salon_short"]:
    # with open("../data/annotations/$_test_rank_full_llava.json".replace("$", dataset), "r") as f:
    #     samples = json.load(f)
    with open("../data/annotations/$_test_rank_clip_llava.json".replace("$", dataset), "r") as f:
        samples = json.load(f)
    
    for exp in experiments[dataset]:
        model_ranks[dataset+"_"+exp] = []

        for sid, sp in enumerate(samples):
            if dataset == "vwp":
                index = sp["scene_full_id"] + "_" + str(sp["story_id"])
            if dataset == "sb20k":
                index = sp["movie_id"] + "_" + str(sp["global_id"])
            if dataset == "salon_short":
                index = sp["portion"] + "_" + sp["sid"]

            for tid, plot in enumerate(sp["narrative"]):
                question = DEFAULT_IMAGE_TOKEN + f'\nStoryline: {plot}'+ f'\nDoes this image fit into the given storyline? Only answer yes or no.'
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                candidate_scores = sp["scores"][tid]

                img_idx = index + "_" + str(tid) + ".jpg"
                img_path = root.replace("$", dataset).replace("*", exp) + img_idx

                image = Image.open(img_path)
                image_tensor = process_images([image], image_processor, llava_model.config)
                image_tensor = [_img.to(dtype=img_dtype, device=device) for _img in image_tensor]
                image_sizes = [image.size]

                out = llava_model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                first_token_scores = out.scores[0][0]
                probs = F.softmax(first_token_scores)
                # target_ids = {"Yes": 9454, "yes": 9693, "YES": 14004}
                score = probs[9454] + probs[9693] + probs[14004]
                score = float(score.detach().cpu().numpy())

                ranking = get_rank(score, candidate_scores)
                model_ranks[dataset+"_"+exp].append(1.0/(ranking+1))  # use MRR
        
final_mrr = {}
for eid, mrrs in model_ranks.items():
    final_mrr[eid] = float(np.mean(mrrs))

with open("../evaluation/mm_interleaved_llava_vqa_mrr_scores.json", "w") as f:
    json.dump(final_mrr, f, indent=2)
