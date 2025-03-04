
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

num_id_map = {"0": 15, "1": 16, "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23, "9": 24}

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

num_align_scores = {}
attr_align_scores = {}
conv_template = "qwen_2"
for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test.json".replace("$", dataset), "r") as f:
        samples = json.load(f)

    for exp in experiments[dataset]:
        num_align_scores[dataset+"_"+exp] = []
        attr_align_scores[dataset+"_"+exp] = []

        for sid, sp in tqdm(enumerate(samples)):
            if dataset == "vwp":
                index = sp["scene_full_id"] + "_" + str(sp["story_id"])
            if dataset == "sb20k":
                index = sp["movie_id"] + "_" + str(sp["global_id"])
            if dataset == "salon_short":
                index = sp["portion"] + "_" + sp["sid"]
            
            sp_char_desc = sp["global_profile"]
            sp_char_pres = [x["present"] for x in sp["scene_characters"]]
            sp_char_num = [int(x["num_present"]) for x in sp["scene_characters"]]
            if len(sp_char_desc) == 0:
                continue
            
            for tid, plot in enumerate(sp["narrative"]):

                turn_char_pres = sp_char_pres[tid]
                turn_char_desc = {}
                for char in turn_char_pres:
                    turn_char_desc[char] = sp_char_desc[char]
                char_count = sp_char_num[tid]
                
                img_idx = index + "_" + str(tid) + ".jpg"
                img_path = root.replace("$", dataset).replace("*", exp) + img_idx
                
                image = Image.open(img_path)
                image_tensor = process_images([image], image_processor, llava_model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
                image_sizes = [image.size]

                question_1 = DEFAULT_IMAGE_TOKEN + f'\nHow many characters are in this image? Only answer an Arabic number.'
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question_1)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                        
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

                if 0 <= char_count <= 9:
                    target_id = num_id_map[str(char_count)]
                    score_1 = probs[target_id]
                else:  # group cases
                    score_1 = None
                    for _, nid in num_id_map.items():
                        if score_1 is None:
                            score_1 = probs[nid]
                        else:
                            score_1 += probs[nid]
                score_1 = float(score_1.detach().cpu().numpy())
                num_align_scores[dataset+"_"+exp].append(score_1)
                
                if char_count > 0 and len(turn_char_desc) > 0:
                    question_2 = DEFAULT_IMAGE_TOKEN + f'\nCharacter descriptions:\n'
                    for name, desp in turn_char_desc.items():
                        question_2 += f'{name}: {desp}\n'
                    question_2 += f'Do characters in this image fit into their descriptions? Only answer yes or no.'
                    
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question_2)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()

                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

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
                    score_2 = probs[9454] + probs[9693] + probs[14004]
                    score_2 = float(score_2.detach().cpu().numpy())
                    attr_align_scores[dataset+"_"+exp].append(score_2)

final_match_num = {}
for eid, score_ls in num_align_scores.items():
    final_match_num[eid] = float(np.mean(score_ls))

final_match_attr = {}
for eid, score_ls in attr_align_scores.items():
    final_match_attr[eid] = float(np.mean(score_ls))

with open("../evaluation/mm_interleaved_llava_vqa_character_num_align_scores.json", "w") as f:
    json.dump(final_match_num, f, indent=2)

with open("../evaluation/mm_interleaved_llava_vqa_character_attr_align_scores.json", "w") as f:
    json.dump(final_match_attr, f, indent=2)
