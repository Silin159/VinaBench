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

karen1 = Image.open('../annot_scripts/karen1.png').convert('RGB')
karen2 = Image.open('../annot_scripts/karen2.png').convert('RGB')
karen3 = Image.open('../annot_scripts/karen3.png').convert('RGB')
karen4 = Image.open('../annot_scripts/karen4.png').convert('RGB')
joseph1 = Image.open('../annot_scripts/joseph1.png').convert('RGB')
joseph2 = Image.open('../annot_scripts/joseph2.png').convert('RGB')
joseph3 = Image.open('../annot_scripts/joseph3.png').convert('RGB')

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

def setting_final(portion="test"):
    
    if os.path.exists("../annotations/"+portion+"_settings_llava.json"):
        with open("../annotations/"+portion+"_settings_llava.json", "r") as f:
            data = json.load(f)
    else:
        with open("../annotations/"+portion+"_clean_final.json", "r") as f:
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
        if "time" not in data[i] and len(data[i]["nar_trans"]) <= 30:
            places = []
            times = []
            for t in range(len(data[i]["nar_trans"])):
                narrative = data[i]["nar_trans"][t]

                image = Image.open("."+data[i]["image_paths"][t]).convert('RGB')
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
                
            data[i]['place'] = places
            data[i]['time'] = times
            # data[i]["style"] = style

            with open("../annotations/"+portion+"_settings_llava.json", "w") as f:
                json.dump(data, f, indent=2)

setting_final(portion="test")
setting_final(portion="train")
