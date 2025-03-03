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

base_messages_characters = [
        {"role": "system",
        "content": "Identify all of the characters in the following narrative. For each character, give the character's name. If the name is not mentioned, give the character's role pronoun (e.g., woman, father) instead. \
                    Only answer with a comma separated list of character names or pronouns. If you are not sure, answer \"do not know\"."},
        {
            "role": "user",
            "content": "Narrative: Karen was cooking lunch on the weekend. She received a call from her friend Elle, inviting her out for lunch."
        },
        {
            "role": "assistant",
            "content": "Karen, Elle"
        },
        {
            "role": "user",
            "content": "Narrative: The bald man gets out of the car, and he is making some fight stance position. \
                        Jeff doesn't know what exactly the bald man is trying to do now."
        },
        {
            "role": "assistant",
            "content": "Jeff, bald man"
        },
    ]


def generate_character_list(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Annotating global character list of {dataset} {portion} set!")

    with open("../annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        if "characters" not in data[i] and len(data[i]["narrative"]) <= 30:
            narrative = " ".join(data[i]["narrative"])
            messages = base_messages_characters + [{"role": "user", "content": f"Narrative: {narrative}"}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            # temperature=0.6,
            # top_p=0.9,
            outputs = model.generate(
                input_ids,
                max_new_tokens=64,
                eos_token_id=terminators,
                do_sample=False,
            )
            response = outputs[0][input_ids.shape[-1]:]
            raw_response = tokenizer.decode(response, skip_special_tokens=True)
            data[i]["characters"] = raw_response.split(", ")

            # for more frequent saving of annotations
            # with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)
    
    with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


generate_character_list(dataset="vwp", portion="test")
generate_character_list(dataset="sb20k", portion="test")
generate_character_list(dataset="salon", portion="test")
generate_character_list(dataset="vwp", portion="train")
generate_character_list(dataset="sb20k", portion="train")
generate_character_list(dataset="salon", portion="train")
