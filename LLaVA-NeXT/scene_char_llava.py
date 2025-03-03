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
joseph3 = Image.open('../data/scripts/examples/joseph3.png').convert('RGB')

base_messages_character_number = [
        {
            "role": "user",
            "content": [karen1, "How many characters are present in the image? Only answer an Arabic number."]
        },
        {
            "role": "assistant",
            "content": ["1"]
        },
        {
            "role": "user",
            "content": [karen4, "How many characters are present in the image? Only answer an Arabic number."]
        },
        {
            "role": "assistant",
            "content": ["2"]
        },
        {
            "role": "user",
            "content": [joseph1, "How many characters are present in the image? Only answer an Arabic number."]
        },
        {
            "role": "assistant",
            "content": ["2"]
        }
    ]

base_messages_character_present_list_options = [
        {
            "role": "user",
            "content": [karen4, "There are 2 characters presented in the image, who are they according to the character list and the narrative context? \
            Answer with a comma separated list of character names or role pronouns. \
            Past Narrative: Karen was cooking lunch on the weekend. She received a call from her friend Elle, inviting her out for lunch. Karen met Elle outside of a restaurant.\
            Narrative: They chose a table to sit down, while Elle read Karen a piece of bad news on the newspaper. \
            Characters: Elle (adult female), Karen (adult female)"]
        },
        {
            "role": "assistant",
            "content": ["Elle, Karen"]
        },
        {
            "role": "user",
            "content": [joseph3, "There is 1 character presented in the image, who is it according to the character list and the narrative context? \
            Answer with a character name or role pronoun. \
            Past Narrative: Jeff is doing a night walk and then he sees a car with a man inside.\
            Narrative: He is going to see who is inside the car. \
            Characters: Joseph (adult male), Jeff (man with long hair)"]
        },
        {
            "role": "assistant",
            "content": ["Jeff"]
        }
    ]


def two_step_characters_present(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Annotating sence-presented characters of {dataset} {portion} set!")
    
    with open("../data/annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)
    
    # present_characters = {}
    conv_template = "qwen_2"  # Make sure you use correct chat template for different models
    
    num_image_tensor = process_images([karen1, karen4, joseph1], image_processor, model.config)
    num_image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in num_image_tensor]
    num_image_sizes = [karen1.size, karen4.size, joseph1.size]
    num_conv = copy.deepcopy(conv_templates[conv_template])
    num_conv.append_message(num_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nHow many characters are present in the image? Only answer an Arabic number.")
    num_conv.append_message(num_conv.roles[1], "1")
    num_conv.append_message(num_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nHow many characters are present in the image? Only answer an Arabic number.")
    num_conv.append_message(num_conv.roles[1], "2")
    num_conv.append_message(num_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nHow many characters are present in the image? Only answer an Arabic number.")
    num_conv.append_message(num_conv.roles[1], "2")

    char_image_tensor = process_images([karen4, joseph3], image_processor, model.config)
    char_image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in char_image_tensor]
    char_image_sizes = [karen4.size, joseph3.size]
    char_conv = copy.deepcopy(conv_templates[conv_template])
    char_conv.append_message(char_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nPast Narrative: Karen was cooking lunch on the weekend. She received a call from her friend Elle, inviting her out for lunch. Karen met Elle outside of a restaurant. \
    Narrative: They chose a table to sit down, while Elle read Karen a piece of bad news on the newspaper. \
    Character List: Elle (adult female), Karen (adult female). \
    There are 2 characters presented in the image, who are they according to the character list and the narrative context? \
    Answer with a comma separated list of character names or role pronouns.")
    char_conv.append_message(char_conv.roles[1], "Elle, Karen")
    char_conv.append_message(char_conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nPast Narrative: Jeff is doing a night walk and then he sees a car with a man inside. \
    Narrative: He is going to see who is inside the car. \
    Character List: Joseph (adult male), Jeff (man with long hair). \
    There is 1 character presented in the image, who is it according to the character list and the narrative context? \
    Answer with a character name or role pronoun.")
    char_conv.append_message(char_conv.roles[1], "Jeff")
    
    for i in range(len(data)):
        if "scene_characters" not in data[i] and len(data[i]["narrative"]) <= 30:
            past_narrative = ""
            scene_list = []
    
            for t in range(len(data[i]["narrative"])):
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
                
                question = DEFAULT_IMAGE_TOKEN + "\nHow many characters are present in the image? Only answer an Arabic number."
                num_conv_prompt = copy.deepcopy(num_conv)
                num_conv_prompt.append_message(num_conv_prompt.roles[0], question)
                num_conv_prompt.append_message(num_conv_prompt.roles[1], None)
                prompt_question = num_conv_prompt.get_prompt()
                final_image_tensor = num_image_tensor + image_tensor
                image_sizes = num_image_sizes + [image.size]
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

                cont = model.generate(
                    input_ids,
                    images=final_image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4,
                )
                num_characters = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                
                narrative = data[i]["narrative"][t]
                
                characters_list = []
                characters_desp = []
                for k, v in data[i]["global_profile"].items():
                    characters_list.append(f"{k} ({v})")
                    characters_desp.append(v)
                
                if len(characters_list) == 0 or all([x == "no description" for x in characters_desp]) or num_characters == "0":
                    scene_list.append({'num_present': num_characters, 'match': num_characters == "0", 'present': []})
                else:
                    characters_list_str = ", ".join(characters_list)
                    
                    if num_characters == "1":
                        present_question = "There is 1 character presented in the image, who is it according to the character list and the narrative context?"
                        note = "Answer with a character name or role pronoun."
                    else:
                        present_question = "There are "+num_characters+" characters presented in the image, who are they according to the character list and the narrative context?"
                        note = "Answer with a comma separated list of character names or role pronouns."

                    question = DEFAULT_IMAGE_TOKEN + f"\nPast Narrative: {past_narrative} Narrative: {narrative} Character List: {characters_list_str}. {present_question} {note}"
                    char_conv_prompt = copy.deepcopy(char_conv)
                    char_conv_prompt.append_message(char_conv_prompt.roles[0], question)
                    char_conv_prompt.append_message(char_conv_prompt.roles[1], None)
                    prompt_question = char_conv_prompt.get_prompt()
                    final_image_tensor = char_image_tensor + image_tensor
                    image_sizes = char_image_sizes + [image.size]
                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

                    cont = model.generate(
                        input_ids,
                        images=final_image_tensor,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=64,
                    )
                    answer = tokenizer.batch_decode(cont, skip_special_tokens=True)

                    present = []
                    for char in answer[0].split(", "):
                        if char in data[i]["global_profile"]:
                            present.append(char)

                    match = int(num_characters) == len(present)
                    
                    scene_list.append({'num_present': num_characters, 'match': match, 'present': present})
                
                past_narrative += " " + narrative
                past_narrative = past_narrative.strip()

            data[i]["scene_characters"] = scene_list
            
            # for more frequent saving of annotations
            # with open("../data/annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)
    
    with open("../data/annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


two_step_characters_present(dataset="vwp", portion="test")
two_step_characters_present(dataset="sb20k", portion="test")
two_step_characters_present(dataset="salon", portion="test")
two_step_characters_present(dataset="vwp", portion="train")
two_step_characters_present(dataset="sb20k", portion="train")
two_step_characters_present(dataset="salon", portion="train")
