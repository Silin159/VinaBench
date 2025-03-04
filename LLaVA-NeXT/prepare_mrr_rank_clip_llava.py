
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
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, torch_dtype=torch_dtype)
model.eval()

for dataset in ["vwp", "sb20k", "salon_short"]:
    with open("../data/annotations/$_test_rank_clip.json".replace("$", dataset), "r") as f:
        samples = json.load(f)

    output_neg = []
    top_k = 100
    conv_template = "qwen_2"
    for sp in tqdm(samples):
        narrative = sp["narrative"]
        if dataset == "vwp":
            sp_neg = {"scene_full_id": sp["scene_full_id"], "story_id": sp["story_id"],
                      "narrative": narrative, "candidates": [], "scores": []}
        if dataset == "sb20k":
            sp_neg = {"movie_id": sp["movie_id"], "global_id": sp["global_id"],
                      "narrative": narrative, "candidates": [], "scores": []}
        if dataset == "salon_short":
            sp_neg = {"portion": sp["portion"], "sid": sp["sid"],
                      "narrative": narrative, "candidates": [], "scores": []}
        
        for tid, plot in enumerate(narrative):
            question = DEFAULT_IMAGE_TOKEN + f'\nStoryline: {plot}'+ f'\nDoes this image fit into the given storyline? Only answer yes or no.'
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            probs = []
            for img_idx in range(len(samples["candidates"])):
                
                image = Image.open(samples["candidates"][img_idx])
                image_tensor = process_images([image], image_processor, model.config)
                image_sizes = [image.size]

                out = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                first_token_scores = out.scores[0]
                assert first_token_scores.shape[0] == 1
                first_token_probs = F.softmax(first_token_scores, dim=1)
                probs.append(first_token_probs.detach().cpu())
            
            all_probs = torch.cat(probs, dim=0)
            # target_ids = {"Yes": 9454, "yes": 9693, "YES": 14004}
            all_scores = all_probs[:,9454] + all_probs[:,9693] + all_probs[:,14004]
            
            topk_scores, topk_ids = torch.topk(all_scores.to(device), k=top_k)
            
            topk_scores_ls = topk_scores.detach().cpu().numpy().tolist()
            topk_ids_ls = topk_ids.detach().cpu().numpy().tolist()
            sp_neg["candidates"].append([samples["candidates"][idx] for idx in topk_ids_ls])
            sp_neg["scores"].append(topk_scores_ls)
        
        output_neg.append(sp_neg)
        
    with open("../data/annotations/$_test_rank_clip_llava.json".replace("$", dataset), "w") as f:
        json.dump(output_neg, f, indent=2)
