
import json
import torch
from PIL import Image
from copy import deepcopy
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

minicpm = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
minicpm = minicpm.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

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
                img = Image.open(img_path).convert('RGB')

                # number
                msgs = []
                char_num_prompt = f'How many characters are in this image? Only answer an Arabic number.'
                msgs.append({'role': 'user', 'content': [img, char_num_prompt]})
                res_num = minicpm.chat(image=None, msgs=msgs, tokenizer=tokenizer,
                                       sampling=False, num_beams=1)
                res_num = res_num.split(" ")[0].strip(".")

                if int(res_num) == char_count:
                    num_align_scores[dataset+"_"+exp].append(1.0)
                else:
                    num_align_scores[dataset+"_"+exp].append(0.0)
                
                # attribute
                if char_count > 0 and len(turn_char_desc) > 0:
                    msgs = []
                    char_attr_prompt = "Character descriptions:\n"
                    for name, desp in turn_char_desc.items():
                        char_attr_prompt += f'{name}: {desp}\n'
                    char_attr_prompt += f'Do characters in this image fit into their descriptions? Only answer yes or no.'
                    
                    msgs.append({'role': 'user', 'content': [img, char_attr_prompt]})
                    res_attr = minicpm.chat(image=None, msgs=msgs, tokenizer=tokenizer,
                                            sampling=False, num_beams=1)
                    
                    if res_attr.lower().startswith("yes"):
                        attr_align_scores[dataset+"_"+exp].append(1.0)
                    else:
                        attr_align_scores[dataset+"_"+exp].append(0.0)

final_match_num = {}
for eid, score_ls in num_align_scores.items():
    final_match_num[eid] = float(np.mean(score_ls))

final_match_attr = {}
for eid, score_ls in attr_align_scores.items():
    final_match_attr[eid] = float(np.mean(score_ls))

with open("./mm_interleaved_minicpm_vqa_character_num_align_scores.json", "w") as f:
    json.dump(final_match_num, f, indent=2)

with open("./mm_interleaved_minicpm_vqa_character_attr_align_scores.json", "w") as f:
    json.dump(final_match_attr, f, indent=2)
