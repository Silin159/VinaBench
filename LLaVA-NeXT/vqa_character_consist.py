
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

char_consist_scores = {}
conv_template = "qwen_2"
for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test.json".replace("$", dataset), "r") as f:
        samples = json.load(f)

    for exp in experiments[dataset]:
        char_consist_scores[dataset+"_"+exp] = []

        for sid, sp in tqdm(enumerate(samples)):
            if dataset == "vwp":
                index = sp["scene_full_id"] + "_" + str(sp["story_id"])
            if dataset == "sb20k":
                index = sp["movie_id"] + "_" + str(sp["global_id"])
            if dataset == "salon_short":
                index = sp["portion"] + "_" + sp["sid"]

            sp_char_desc = sp["global_profile"]
            sp_char_pres = [x["present"] for x in sp["scene_characters"]]
            if len(sp_char_desc) == 0:
                continue

            consist_score = 0.0
            char_eval_count = 0
            for char, desp in sp_char_desc.items():
                appear_tids = []
                for tid in range(len(sp["narrative"])):
                    if char in sp_char_pres[tid]:
                        appear_tids.append(tid)
                
                if len(appear_tids) < 2:
                    pass
                else:
                    images = []
                    image_sizes = []
                    question = ""
                    counter = 1
                    for tid in appear_tids:
                        img_idx = index + "_" + str(tid) + ".jpg"
                        img_path = root.replace("$", dataset).replace("*", exp) + img_idx
                        img = Image.open(img_path)
                        images.append(img)
                        image_sizes.append(img.size)
                        question += f'Image {counter}: {DEFAULT_IMAGE_TOKEN}\n'
                        counter += 1
                    
                    image_tensors = process_images(images, image_processor, llava_model.config)
                    image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
                    
                    question += f'Do all these images contain the same charcater {char}: {desp}? Only answer yes or no.'
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    
                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                            
                    out = llava_model.generate(
                        inputs=input_ids,
                        images=image_tensors,
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
                    
                    consist_score += score
                    char_eval_count += 1
                    
            if char_eval_count > 0:
                char_consist_scores[dataset+"_"+exp].append(consist_score/char_eval_count)

final_scores = {}
for eid, score_ls in char_consist_scores.items():
    final_scores[eid] = float(np.mean(score_ls))

with open("../evaluation/mm_interleaved_llava_vqa_char_consist_scores.json", "w") as f:
    json.dump(final_scores, f, indent=2)
