from copy import deepcopy
import json
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from transformers.image_utils import load_image
import sys

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# your huggingface access token
access_token = ""
os.environ["HF_ACCESS_TOKEN"] = access_token

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

base_messages_character_descriptions = [
        {"role": "system", "content": "You are given a narrative and a character. Using the narrative, \
        give some phrases to physically describe the character, which can include their age range, gender, \
        social role and other sustained physical features that the narrative mentions. \
        Do not give more information than you can infer from the narrative."},
        {
            "role": "user",
            "content": "Narrative: Karen was cooking lunch on the weekend. She received a call from her friend Elle, inviting her out for lunch. \
            Character: Karen"
        },
        {
            "role": "assistant",
            "content": "adult female"
        },
        {
            "role": "user",
            "content": "Narrative: Joseph gets out of the car, and he is making some fight stance position. \
            Jeff doesn't know what exactly Joseph is trying to do now. \
            Character: Joseph"
        },
        {
            "role": "assistant",
            "content": "adult male"
        },
        {
            "role": "user",
            "content": "Narrative: A family goes to the store to buy milk. They cannot find any milk in the store, \
            so Kate drove her son back home. \
            Character: son"
        },
        {
            "role": "assistant",
            "content": "young boy, Kate's son"
        }
    ]


def generate_character_attr(portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Annotating global character attributes of {dataset} {portion} set!")

    with open("../annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        if "global_profile" not in data[i] and len(data[i]["narrative"]) <= 30:
            characters = data[i]['characters']
            narratives = data[i]['narrative']
            
            narrative = " ".join(narratives)
            profile = {}
            for character in characters:
                if character.lower().strip() != "do not know":
                    messages = base_messages_character_descriptions + [{"role": "user", "content": f"Narrative: {narrative} Character: {character}"}]
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)
                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=64,
                        eos_token_id=terminators,
                        do_sample=False,
                    )
                    response = outputs[0][input_ids.shape[-1]:]
                    answer = tokenizer.decode(response, skip_special_tokens=True)
                    if any([x in answer.lower() for x in ["no mention", "no character", "no information"]]):
                        answer = "no description"
                    profile[character] = answer
                
            data[i]['global_profile'] = profile
            data[i].pop('characters')

            # for more frequent saving of annotations
            # with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)
    
    with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


generate_character_attr(dataset="vwp", portion="test")
generate_character_attr(dataset="sb20k", portion="test")
generate_character_attr(dataset="salon", portion="test")
generate_character_attr(dataset="vwp", portion="train")
generate_character_attr(dataset="sb20k", portion="train")
generate_character_attr(dataset="salon", portion="train")
