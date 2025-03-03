import os
import json
# import wget
# import pandas as pd
# import requests
import torch
from PIL import Image
from io import BytesIO
import nltk

from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
# import sys
# sys.path.append("/mnt/nlp/home/lmi/")
# import annotation_ids

processor = AutoProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Idefics2") # do_image_splitting is False by default
model = AutoModelForVision2Seq.from_pretrained(
    "TIGER-Lab/Mantis-8B-Idefics2",
    device_map="auto"
)
generation_kwargs = {
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}

base_messages = [
    {"role": "system", "content": "You are given an image and a corresponding narrative that tells a story about the image. Please describe the image \
     in detail in two or three sentences."},
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Narrative: Karen was cooking lunch at home on a weekend."},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "A woman in a green shirt is standing in a kitchen, washing dishes in a sink. The kitchen is well-equipped with a stove, oven, and various kitchen utensils. There are multiple cups and bowls on the counter, and a vase can be seen on the counter as well."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Narrative: They chose a table to sit down, while Elle read Karen a piece of bad news on the newspaper."},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Two women are sitting at a table in a restaurant. One woman is wearing a pink shirt and the other is wearing a white shirt. The woman wearing a pink shirt is holding a newspaper and appears to be engaged in reading."},
        ]
    }
    ]
image_shot1 = Image.open("./examples/karen1.png").convert('RGB')
image_shot2 = Image.open("./examples/karen4.png").convert('RGB')


def caption_trancate(caption, max_length=100):
    tokenized_cap = nltk.word_tokenize(caption)
    if len(tokenized_cap) > max_length:
        tokenized_cap = tokenized_cap[:max_length]
    return " ".join(tokenized_cap)


def generate_captions(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Annotating image captions of {dataset} {portion} set!")

    messages = base_messages
    # read annotations
    with open("../annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)
    
    for i in range(len(data)):
        # item = data[i]
        if "captions" not in data[i] and len(data[i]["narrative"]) <= 30:
            if dataset == "vwp":
                frame_links = data[i]["image_links"]
                frames = []
                for img_link in frame_links:
                    out_pth = "../images/"+img_link.split("/")[-2]
                    img_file = out_pth+"/"+img_link.split("/")[-1]
                    # if not os.path.exists(img_file):
                    #     os.makedirs(out_pth, exist_ok=True)
                    #     wget.download(img_link, out=out_pth)
                    frames.append(img_file)
            elif dataset == "sb20k":
                frames = ["../storyboard20k/frames/"+portion+"/"+pth for pth in data[i]["key_frames"]]
            elif dataset == "salon":
                frames = ["."+pth for pth in data[i]["image_paths"]]
            else:
                raise NotImplementedError
            
            narratives = data[i]["narrative"]
            outputs = []
            for k in range(len(frames)):
                frame = frames[k]
                narrative = narratives[k]
                messages = base_messages + [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Narrative: {narrative}"},
                    ]
                }]
                image = Image.open(frame).convert('RGB')
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[image_shot1, image_shot2, image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                generated_ids = model.generate(**inputs, **generation_kwargs)
                response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                outputs.append(caption_trancate(response[0]))
            
            data[i]["captions"] = outputs

            # for more frequent saving of annotations
            # with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)
    
    with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


generate_captions(dataset="vwp", portion="test")
generate_captions(dataset="sb20k", portion="test")
generate_captions(dataset="salon", portion="test")
generate_captions(dataset="vwp", portion="train")
generate_captions(dataset="sb20k", portion="train")
generate_captions(dataset="salon", portion="train")
