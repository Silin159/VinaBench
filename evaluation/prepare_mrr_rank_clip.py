
import json
from tqdm import tqdm
import numpy as np
import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from copy import deepcopy
from transformers import CLIPProcessor, CLIPModel
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image_pools = {"vwp": "../data/annotations/images",
               "sb20k": "../data/annotations/storyboard20k",
               "salon_short": "../data/annotations/Image_inpainted"}

def link_to_file(img_link, dataset):
    if dataset == "vwp":
        out_pth = os.path.join(image_pools[dataset], img_link.split("/")[-2])
        img_file = os.path.join(out_pth, img_link.split("/")[-1])
        # os.makedirs(out_pth, exist_ok=True)
        # if not os.path.exists(img_file):
        #     wget.download(img_link, out=out_pth)
        return img_file
    elif dataset == "sb20k":
        return os.path.join(image_pools[dataset], img_link)
    elif dataset == "salon_short":
        return os.path.join(image_pools[dataset], "/".join(img_link.split("/")[2:]))
    else:
        raise ValueError

for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test.json".replace("$", dataset), "r") as f:
        samples = json.load(f)
    
    # image embedding
    visited = set()
    img_ids = []
    img_embs = []
    
    for sp in tqdm(samples):
        if dataset == "vwp":
            image_links = sp["image_links"]
        if dataset == "sb20k":
            image_links = sp["key_frames"]
        if dataset == "salon_short":
            image_links = sp["image_paths"]
        
        for img_link in image_links:
            img_file = link_to_file(img_link, dataset)
            if not img_file in visited:
                img = Image.open(img_file)
                inputs = processor(images=img, return_tensors="pt").to(device)
                img_embed = model.get_image_features(**inputs)
                
                visited.add(img_file)
                # save_file = img_file.replace(".jpg", ".npy")
                # np.save(save_file, img_embed.cpu().detach().numpy())
                img_ids.append(img_file)
                img_embs.append(img_embed)

    output_neg = []
    top_k = 100
    image_batch_size = 1024
    img_total = len(img_ids)
    
    for sp in tqdm(samples):
        if dataset == "vwp":
            sp_neg = {"scene_full_id": sp["scene_full_id"], "story_id": sp["story_id"]}
        if dataset == "sb20k":
            sp_neg = {"movie_id": sp["movie_id"], "global_id": sp["global_id"]}
        if dataset == "salon_short":
            sp_neg = {"portion": sp["portion"], "sid": sp["sid"]}
        
        narrative = sp["narrative"]
        sp_neg["narrative"] = narrative

        len_nar = len(narrative)
        nar_inputs = processor(text=narrative, return_tensors="pt", padding=True).to(device)
        nar_emb = model.get_text_features(**nar_inputs)

        topk_scores = torch.zeros(len_nar, top_k).to(device)
        topk_ids = torch.zeros(len_nar, top_k).long().to(device)

        for shift_img in range(0, img_total, image_batch_size):
            right_img = min(shift_img+image_batch_size, img_total)
            img_batch_emb = torch.cat(img_embs[shift_img:right_img], dim=0).to(device)

            similarity = pairwise_cosine_similarity(nar_emb, img_batch_emb)
            topk_sim, topk_sim_idx = torch.topk(similarity, k=top_k, dim=1)
            topk_sim_idx = topk_sim_idx + shift_img

            topk_cand_scores = torch.cat([topk_scores, topk_sim], dim=1)
            topk_cand_idx = torch.cat([topk_ids, topk_sim_idx], dim=1)
            topk_sim_new, topk_idx_select = torch.topk(topk_cand_scores, k=top_k, dim=1)

            topk_scores = topk_sim_new
            for i in range(len_nar):
                topk_ids[i] = topk_cand_idx[i][topk_idx_select[i]]
        
        topk_scores_ls = topk_scores.detach().cpu().numpy().tolist()
        topk_ids_ls = topk_ids.detach().cpu().numpy().tolist()
        matches = [[img_ids[idx] for idx in img_match] for img_match in topk_ids_ls]
        match_scores = topk_scores_ls
        
        sp_neg["candidates"] = matches
        sp_neg["scores"] = match_scores
        output_neg.append(sp_neg)
    
    with open("../data/annotations/$_test_rank_clip.json".replace("$", dataset), "w") as f:
        json.dump(output_neg, f, indent=2)
