
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

char_consist_scores = {}
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
                    content = []
                    for tid in appear_tids:
                        img_idx = index + "_" + str(tid) + ".jpg"
                        img_path = root.replace("$", dataset).replace("*", exp) + img_idx
                        img = Image.open(img_path).convert('RGB')
                        content.append(img)
                    
                    consist_prompt = f'Do all these images contain the same charcater {char}: {desp}? Only answer yes or no.'
                    content.append(consist_prompt)

                    msgs = [{'role': 'user', 'content': content}]
                    res = minicpm.chat(image=None, msgs=msgs, tokenizer=tokenizer,
                                       sampling=False, num_beams=1)
                    res = res.split(" ")[0].strip(".")

                    if res.lower().startswith("yes"):
                        consist_score += 1.0
                    else:
                        consist_score += 0.0
                    char_eval_count += 1
            
            if char_eval_count > 0:
                char_consist_scores[dataset+"_"+exp].append(consist_score/char_eval_count)

final_scores = {}
for eid, score_ls in char_consist_scores.items():
    final_scores[eid] = float(np.mean(score_ls))

with open("./mm_interleaved_minicpm_vqa_char_consist_scores.json", "w") as f:
    json.dump(final_scores, f, indent=2)
