import os
import json
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

from mm_interleaved.utils.clip_sim_score import tensor_to_pil, calculate_clip_sim_i2i
from mm_interleaved.utils.fid_score import calculate_fid_given_paths

# take our experiments on MM-Interleaved as examples
gold_root = "./OUTPUT/$_inf/*/gold/
pred_root = "./OUTPUT/$_inf/*/pred/"
experiments = {"vwp": ["no_cons_ds250_full_inf_32k",
                       "llama_cons_ds250_full_inf_30k",
                       "gold_cons_ds250_full_inf_30k"],
               "sb20k": ["no_cons_ds250_full_inf_32k",
                         "llama_cons_ds250_full_inf_30k",
                         "gold_cons_ds250_full_inf_30k"],
               "salon_short": ["no_cons_ds250_full_inf_32k",
                               "llama_cons_ds250_full_inf_36k",
                               "gold_cons_ds250_full_inf_36k"]}

metrics = {}
for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test.json".replace("$", dataset), "r") as f:
        samples = json.load(f)
    for exp in experiments[dataset]:
        metrics[dataset+"_"+exp] = {}
        eval_image_pairs = []
        eval_image_id = 0

        for sid, sp in tqdm(enumerate(samples)):
            if dataset == "vwp":
                index = sp["scene_full_id"] + "_" + str(sp["story_id"])
            if dataset == "sb20k":
                index = sp["movie_id"] + "_" + str(sp["global_id"])
            if dataset == "salon_short":
                index = sp["portion"] + "_" + sp["sid"]
            
            for tid, plot in enumerate(sp["narrative"]):
                
                img_idx = index + "_" + str(tid) + ".jpg"
                pred_img_pth = pred_root.replace("$", dataset).replace("*", exp) + img_idx
                gold_img_pth = gold_root.replace("$", dataset).replace("*", exp) + img_idx

                img_result = {"story_id": index, "sample_idx": tid, "image_id": eval_image_id,
                              "image_path": pred_img_pth, "image_gt_path": gold_img_pth}
                eval_image_pairs.append(img_result)
                eval_image_id += 1
    
        clip_score = calculate_clip_sim_i2i(eval_image_pairs, device="cuda", batch_size=256)
        gt_paths = [r["image_gt_path"] for r in eval_image_pairs]
        pred_paths = [r["image_path"] for r in eval_image_pairs]
        fid = calculate_fid_given_paths((gt_paths, pred_paths))
        metrics[dataset+"_"+exp]["FID"] = fid
        metrics[dataset+"_"+exp]["CLIP-I"] = float(clip_score.detach().cpu().numpy())

with open("../evaluation/mm_interleaved_fid_clipi_scores.json", "w") as wf:
    json.dump(metrics, wf, indent=2)
