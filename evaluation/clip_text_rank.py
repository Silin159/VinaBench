import json
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from transformers import CLIPProcessor, CLIPModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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
model_clip_t = {}
for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test_rank_clip.json".replace("$", dataset), "r") as f:
        samples = json.load(f)
    
    for exp in experiments[dataset]:
        model_ranks[dataset+"_"+exp] = []
        model_clip_t[dataset+"_"+exp] = []

        for sid, sp in tqdm(enumerate(samples)):
            if dataset == "vwp":
                index = sp["scene_full_id"] + "_" + str(sp["story_id"])
            if dataset == "sb20k":
                index = sp["movie_id"] + "_" + str(sp["global_id"])
            if dataset == "salon_short":
                index = sp["portion"] + "_" + sp["sid"]

            for tid, plot in enumerate(sp["narrative"]):
                plot_input = processor(text=[plot], return_tensors="pt", padding=True).to(device)
                plot_emb = clip_model.get_text_features(**plot_input)
                candidate_scores = sp["scores"][tid]

                img_idx = index + "_" + str(tid) + ".jpg"
                img_pth = root.replace("$", dataset).replace("*", exp) + img_idx

                img = Image.open(img_pth)
                img_input = processor(images=[img], return_tensors="pt").to(device)
                img_embed = clip_model.get_image_features(**img_input)
                
                txt_similarity = pairwise_cosine_similarity(plot_emb, img_embed)
                clip_t_score = txt_similarity.detach().cpu().numpy().tolist()[0][0]

                ranking = get_rank(clip_t_score, candidate_scores)
                model_ranks[dataset+"_"+exp].append(1.0/(ranking+1))  # use MRR
                model_clip_t[dataset+"_"+exp].append(clip_t_score)

for eid, mrrs in model_ranks.items():
    final_results[eid] = {}
    final_results[eid]["mrr_rank"] = float(np.mean(mrrs))

for eid, clip_ts in model_clip_t.items():
    final_results[eid]["clip_t"] = float(np.mean(clip_ts))

with open("./mm_interleaved_clipt_mrr_scores.json", "w") as f:
    json.dump(final_results, f, indent=2)
