
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

loc_consist_scores = {}
for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test.json".replace("$", dataset), "r") as f:
        samples = json.load(f)

    for exp in experiments[dataset]:
        loc_consist_scores[dataset+"_"+exp] = []

        for sid, sp in tqdm(enumerate(samples)):
            if dataset == "vwp":
                index = sp["scene_full_id"] + "_" + str(sp["story_id"])
            if dataset == "sb20k":
                index = sp["movie_id"] + "_" + str(sp["global_id"])
            if dataset == "salon_short":
                index = sp["portion"] + "_" + sp["sid"]

            loc_set = list(set(sp["location"]))

            consist_score = 0.0
            loc_eval_count = 0
            for loc in loc_set:
                target_tids = []
                for tid in range(len(sp["narrative"])):
                    if sp["location"][tid] == loc:
                        target_tids.append(tid)
            
                if len(target_tids) < 2:
                    pass
                else:
                    content = []
                    for tid in target_tids:
                        img_idx = index + "_" + str(tid) + ".jpg"
                        img_path = root.replace("$", dataset).replace("*", exp) + img_idx
                        img = Image.open(img_path).convert('RGB')
                        content.append(img)
                    
                    consist_prompt = f'Are all these images taken at the same {loc}? Only answer yes or no.'
                    content.append(consist_prompt)

                    msgs = [{'role': 'user', 'content': content}]
                    res = minicpm.chat(image=None, msgs=msgs, tokenizer=tokenizer,
                                       sampling=False, num_beams=1)
                    res = res.split(" ")[0].strip(".")

                    if res.lower().startswith("yes"):
                        consist_score += 1.0
                    else:
                        consist_score += 0.0
                    loc_eval_count += 1
            
            if loc_eval_count > 0:
                loc_consist_scores[dataset+"_"+exp].append(consist_score/loc_eval_count)          

final_scores = {}
for eid, score_ls in loc_consist_scores.items():
    final_scores[eid] = float(np.mean(score_ls))

with open("./mm_interleaved_minicpm_vqa_location_consist_scores.json", "w") as f:
    json.dump(final_scores, f, indent=2)
