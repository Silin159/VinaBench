import json
import requests
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for flash_attn on MiniCPM 2.6 code example"""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

karen1 = Image.open('./examples/karen1.png').convert('RGB')
karen2 = Image.open('./examples/karen2.png').convert('RGB')
karen3 = Image.open('./examples/karen3.png').convert('RGB')
karen4 = Image.open('./examples/karen4.png').convert('RGB')
'''
joseph1 = Image.open('./joseph1.png').convert('RGB')
joseph2 = Image.open('./joseph2.png').convert('RGB')
joseph3 = Image.open('./joseph3.png').convert('RGB')

base_messages_setting = [
        {
            "role": "user",
            "content": [karen1, "Identify the setting of the image, where the the following narrative takes place. Narrative: Karen was cooking lunch on the weekend."]
        },
        {
            "role": "assistant",
            "content": ["kitchen"]
        },
        {
            "role": "user",
            "content": [karen4, "Identify the setting of the image, where the the following narrative takes place. Narrative: They chose a table to sit down, while Elle told Karen a piece of bad news on the newspaper."]
        },
        {
            "role": "assistant",
            "content": ["restaurant"]
        }
    ]

base_messages_time = [
        {
            "role": "user",
            "content": [karen1, "Use the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning, afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: Karen was cooking lunch on the weekend."]
        },
        {
            "role": "assistant",
            "content": ["morning"]
        },
        {
            "role": "user",
            "content": [karen4, "Use the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning, afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: Elle read Karen a piece of bad news on the newspaper at afternoon tea."]
        },
        {
            "role": "assistant",
            "content": ["afternoon"]
        },
        {
            "role": "user",
            "content": [joseph1, "Use the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning, afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: Joseph gets out of the car, and he is making some fight stance position."]
        },
        {
            "role": "assistant",
            "content": ["unclear"]
        },
    ]
'''
base_messages_style = [
        {
            "role": "user",
            "content": [karen1, karen2, karen3, karen4, "Identify the style of the images accompanying the following narrative. Your answer must be one of the following choices: photorealistic, fantasy art, digital art, pop art, comic book, cartoon, surrealist, black and white photographic. Narrative: Karen was cooking lunch on the weekend. She received a call from her friend Elle, inviting her out for lunch. Karen met Elle outside of a restaurant. They chose a table to sit down, while Elle told Karen a piece of bad news on the newspaper."]
        },
        {
            "role": "assistant",
            "content": ["photorealistic"]
        }
    ]


def scene_setting(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Annotating global image style of {dataset} {portion} set!")
    
    with open("../annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        if "style" not in data[i] and len(data[i]["narrative"]) <= 30:
            narratives = data[i]['narrative']
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
            
            '''We use LLaVa-OneVision instead of MiniCPM for annotating time and location
            places = []
            times = []
            for k in range(len(frames)):
                frame = frames[k]
                narrative = narratives[k]
                image = Image.open(frame).convert('RGB')
                narrative = narratives[k]
                question = f"Identify the setting of the image, where the the following narrative takes place. Narrative: {narrative}"
                question2 = f"Use the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning,\
                    afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: {narrative}"
                messages = base_messages_setting + [{"role": "user", "content": [image, question]}]
                setting = model.chat(
                    image=None,
                    msgs=messages,
                    tokenizer=tokenizer
                )
                messages = base_messages_time + [{"role": "user", "content": [image, question2]}]
                time = model.chat(
                    image=None,
                    msgs=messages,
                    tokenizer=tokenizer
                )
                places.append(setting)
                times.append(time)
            '''

            if len(frames) > 4:
                frames = frames[0:4]
            images = []
            for frame in frames:
                images.append(Image.open(frame).convert('RGB'))
            narrative = " ".join(narratives[0:4])
            question = f"Identify the style of the images accompanying the following narrative. Your answer must be one of the following choices: photorealistic, fantasy art, digital art, pop art, comic book, cartoon, surrealist, black and white photographic. Narrative: {narrative}"
            messages = base_messages_style + [{"role": "user", "content": images + [question]}]
            style = model.chat(
                image=None,
                msgs=messages,
                tokenizer=tokenizer
            )
            
            # data[i]['place'] = places
            # data[i]['time'] = times

            stys = []
            for sty in ["photorealistic", "fantasy art", "digital art", "pop art", "comic book", "cartoon", "surrealistic", "black and white"]:
                if sty in style:
                    stys.append(sty)
            
            if len(stys) != 1:
                data[i]["style"] = "not unified"
            else:
                data[i]["style"] = stys[0]

            # for more frequent saving of annotations
            # with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)
    
    with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


scene_setting(dataset="vwp", portion="test")
scene_setting(dataset="sb20k", portion="test")
scene_setting(dataset="salon", portion="test")
scene_setting(dataset="vwp", portion="train")
scene_setting(dataset="sb20k", portion="train")
scene_setting(dataset="salon", portion="train")
