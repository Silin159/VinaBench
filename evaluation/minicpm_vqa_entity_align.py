
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

entity_align_scores = {}
for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test.json".replace("$", dataset), "r") as f:
        samples = json.load(f)

    for exp in experiments[dataset]:
        entity_align_scores[dataset+"_"+exp] = []
    
        for sid, sp in tqdm(enumerate(samples)):
            if dataset == "vwp":
                index = sp["scene_full_id"] + "_" + str(sp["story_id"])
            if dataset == "sb20k":
                index = sp["movie_id"] + "_" + str(sp["global_id"])
            if dataset == "salon_short":
                index = sp["portion"] + "_" + sp["sid"]

            for tid, plot in enumerate(sp["narrative"]):
                entities = sp["linked_entities"][tid].split("; ")

                img_idx = index + "_" + str(tid) + ".jpg"
                img_path = root.replace("$", dataset).replace("*", exp) + img_idx
                img = Image.open(img_path).convert('RGB')
                
                match_count = 0
                total_count = 0
                for ent in entities:
                    msgs = []
                    # skip character name entity
                    if len(ent) == 0 or ent[0].isupper():
                        pass
                    else:
                        ent_prompt = f'Does this image contain or imply \'{ent}\'? Only answer yes or no.'
                        msgs.append({'role': 'user', 'content': [img, ent_prompt]})
                        res = minicpm.chat(image=None, msgs=msgs, tokenizer=tokenizer,
                                           sampling=False, num_beams=1)

                        total_count += 1
                        if res.lower().startswith("yes"):
                            match_count += 1
                
                if total_count > 0:
                    entity_align_scores[dataset+"_"+exp].append(float(match_count)/total_count)

final_match = {}
for eid, score_ls in entity_align_scores.items():
    final_match[eid] = float(np.mean(score_ls))

with open("./mm_interleaved_minicpm_vqa_entity_align_scores.json", "w") as f:
    json.dump(final_match, f, indent=2)
