from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import os
import sys
import warnings
import json
from tqdm import tqdm

warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov-sft"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None)  # Add any other thing you want to pass in llava_model_args

model.eval()

karen1 = Image.open('../data/scripts/examples/karen1.png').convert('RGB')
karen4 = Image.open('../data/scripts/examples/karen4.png').convert('RGB')
joseph1 = Image.open('../data/scripts/examples/joseph1.png').convert('RGB')

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

def time_location_setting(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Annotating sence time and location settings of {dataset} {portion} set!")

    with open("../data/annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)
    
    conv_template = "qwen_2"

    loc_image_tensor = process_images([karen1, karen4], image_processor, model.config)
    loc_image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in loc_image_tensor]
    loc_image_sizes = [karen1.size, karen4.size]
    loc_conv = copy.deepcopy(conv_templates[conv_template])
    loc_conv.append_message(loc_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nIdentify the setting of the image, where the the following narrative takes place. Narrative: Karen was cooking lunch on the weekend.")
    loc_conv.append_message(loc_conv.roles[1], "kitchen")
    loc_conv.append_message(loc_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nIdentify the setting of the image, where the the following narrative takes place. Narrative: They chose a table to sit down, while Elle told Karen a piece of bad news on the newspaper.")
    loc_conv.append_message(loc_conv.roles[1], "restaurant")

    time_image_tensor = process_images([karen1, karen4, joseph1], image_processor, model.config)
    time_image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in time_image_tensor]
    time_image_sizes = [karen1.size, karen4.size, joseph1.size]
    time_conv = copy.deepcopy(conv_templates[conv_template])
    time_conv.append_message(time_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nUse the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning, afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: Karen was cooking lunch on the weekend.")
    time_conv.append_message(time_conv.roles[1], "morning")
    time_conv.append_message(time_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nUse the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning, afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: Elle read Karen a piece of bad news on the newspaper at afternoon tea.")
    time_conv.append_message(time_conv.roles[1], "afternoon")
    time_conv.append_message(time_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nUse the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning, afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: Joseph gets out of the car, and he is making some fight stance position.")
    time_conv.append_message(time_conv.roles[1], "unclear")
    
    for i in range(len(data)):
        if "time" not in data[i] and len(data[i]["narrative"]) <= 30:
            places = []
            times = []
            for t in range(len(data[i]["narrative"])):
                narrative = data[i]["narrative"][t]

                if dataset == "vwp":
                    img_link = data[i]["image_links"][t]
                    out_pth = "../data/images/"+img_link.split("/")[-2]
                    img_file = out_pth+"/"+img_link.split("/")[-1]
                    # if not os.path.exists(img_file):
                    #     os.makedirs(out_pth, exist_ok=True)
                    #     wget.download(img_link, out=out_pth)
                elif dataset == "sb20k":
                    img_file = "../data/storyboard20k/frames/"+portion+"/"+data[i]["key_frames"][t]
                elif dataset == "salon":
                    img_file = "."+data[i]["image_paths"][t]
                else:
                    raise NotImplementedError
                
                image = Image.open(img_file).convert('RGB')

                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

                question = DEFAULT_IMAGE_TOKEN + f"\nIdentify the setting of the image, where the the following narrative takes place. Narrative: {narrative}"
                loc_conv_prompt = copy.deepcopy(loc_conv)
                loc_conv_prompt.append_message(loc_conv_prompt.roles[0], question)
                loc_conv_prompt.append_message(loc_conv_prompt.roles[1], None)
                prompt_question = loc_conv_prompt.get_prompt()
                final_image_tensor = loc_image_tensor + image_tensor
                image_sizes = loc_image_sizes + [image.size]
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                cont = model.generate(
                    input_ids,
                    images=final_image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=16,
                )
                loc = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

                question2 = DEFAULT_IMAGE_TOKEN + f"Use the image to identify the time of day during which the following narrative takes place. Your answer must be one of the following choices: early morning, morning,\
                    afternoon, evening, night. If the time of day is unclear in the image and narrative, answer unclear. Narrative: {narrative}"
                time_conv_prompt = copy.deepcopy(time_conv)
                time_conv_prompt.append_message(time_conv_prompt.roles[0], question2)
                time_conv_prompt.append_message(time_conv_prompt.roles[1], None)
                prompt_question = time_conv_prompt.get_prompt()
                final_image_tensor = time_image_tensor + image_tensor
                image_sizes = time_image_sizes + [image.size]
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                cont = model.generate(
                    input_ids,
                    images=final_image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=16,
                )
                time = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

                places.append(loc)
                times.append(time)
                
            data[i]["time"] = []
            for tim in times:
                if tim in ["early morning", "morning", "afternoon", "evening", "night"]:
                    data[i]["time"].append(tim)
                else:
                    data[i]["time"].append("unclear")
            
            loc_filt_prefix = [" is likely*", " is appears to be*", " is*"]
            article = [" in an ", " in a ", " in the ", " in ",
                       " on an ", " on a ", " on the ", " on ",
                       " at an ", " at a ", " at the ", " at ",
                       " an ", " a ", " the ", " "]
            loc_ls = [x.split(",")[0].split(" where ")[0].strip(".") for x in places]
            
            data[i]["location"] = []
            for lc in loc_ls:
                if " is not " in lc or "unclear" in lc:
                    data[i]["location"].append("unclear")
                else:
                    for pf in loc_filt_prefix:
                        for art in article:
                            prefix = pf.replace("*", art)
                            if prefix in lc:
                                lc = lc.split(prefix)[1]
                    data[i]["location"].append(lc)
            
            # for more frequent saving of annotations
            # with open("../data/annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)
    
    with open("../data/annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


time_location_setting(dataset="vwp", portion="test")
time_location_setting(dataset="sb20k", portion="test")
time_location_setting(dataset="salon", portion="test")
time_location_setting(dataset="vwp", portion="train")
time_location_setting(dataset="sb20k", portion="train")
time_location_setting(dataset="salon", portion="train")
