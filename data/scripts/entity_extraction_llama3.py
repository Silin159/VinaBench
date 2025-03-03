from copy import deepcopy
import json
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from transformers.image_utils import load_image
import sys
import nltk
# from nltk.corpus import wordnet as wn
# import nltk
# import spacy

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

base_messages_actions = [
    {"role": "system", "content": "You are given a caption. Output a list of verbs or verb phrases that are actions in the caption.\
        If there is no verb or verb phrase that belongs to actions, report none."},
    {
        "role": "user",
        "content": "Caption: She is cooking lunch in the kitchen with the milk she bought from the store."
    },
    {
        "role": "assistant",
        "content": "cooking, bought"
    },
    {
        "role": "user",
        "content": "Caption: He should wait before going swimming, but instead he will hike with his friends."
    },
    {
        "role": "assistant",
        "content": "wait, going swimming, hike"
    },
    {
        "role": "user",
        "content": "Caption: It was a beautiful sunny day."
    },
    {
        "role": "assistant",
        "content": "none"
    }
]

base_messages_key_entity = [
    {"role": "system", "content": "You are given a caption. Output a list of nouns or noun phrases that are people in the caption. \
    If there is no noun or noun phrase that belongs to people, report none."},
    {
        "role": "user",
        "content": "Caption: A woman in a green shirt is standing in a kitchen, washing dishes in a sink. The kitchen is well-equipped with a stove, oven, and various kitchen utensils. There are multiple cups and bowls on the counter, and a vase can be seen on the counter as well."
    },
    {
        "role": "assistant",
        "content": "woman in a green shirt"
    },
    {
        "role": "user",
        "content": "Caption: A teacher is smiling to a group of students in front of a public phone. The teacher talks to the student's family."
    },
    {
        "role": "assistant",
        "content": "teacher, group of students, student's family"
    },
    {
        "role": "user",
        "content": "Caption: The image is of a winter scene with barren trees, snow on the ground, and a few buildings in the background."
    },
    {
        "role": "assistant",
        "content": "none"
    }
    ]

base_messages_non_key_entity = [
    {"role": "system", "content": "You are given a caption. Output a list of nouns or noun phrases that are non-human objects in the caption. \
    If there is no noun or noun phrase that belongs to non-human objects, report none."},
    {
        "role": "user",
        "content": "Caption: A woman in a green shirt is standing in a kitchen, washing dishes in a sink. The kitchen is well-equipped with a stove, oven, and various kitchen utensils. There are multiple cups and bowls on the counter, and a vase can be seen on the counter as well."
    },
    {
        "role": "assistant",
        "content": "green shirt, kitchen, dishes, sink, stove, oven, kitchen utensils, cups, bowls, counter, vase"
    },
    {
        "role": "user",
        "content": "Caption: A teacher is smiling to a group of students in front of a public phone. The teacher talks to the student's family."
    },
    {
        "role": "assistant",
        "content": "public phone"
    },
    {
        "role": "user",
        "content": "Caption: The image is of three people having a conversation."
    },
    {
        "role": "assistant",
        "content": "none"
    }
    ]


def filter_entities(raw_entity_list):
    entity_list = []
    for raw_entity in raw_entity_list:
        ents = []
        for ent in raw_entity.split(", "):
            if ent != "none":
                ents.append(ent) 
        entity_list.append(", ".join(ents))
    return entity_list


def prompt_llama_entity(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Extracting image (caption) entities of {dataset} {portion} set!")
    
    with open("../annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        captions = data[i]['captions']
        
        if 'key' not in data[i] and len(data[i]["narrative"]) <= 30:
            key_entities = []
            non_key_entities = []
            actions = []

            for idx, caption in enumerate(captions):
                messages = base_messages_key_entity + [{"role": "user", "content": f"Caption: {caption}"}]
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
                key_entities.append(tokenizer.decode(response, skip_special_tokens=True))

                messages = base_messages_non_key_entity + [{"role": "user", "content": f"Caption: {caption}"}]
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
                non_key_entities.append(tokenizer.decode(response, skip_special_tokens=True))

                messages = base_messages_actions + [{"role": "user", "content": f"Caption: {caption}"}]
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
                actions.append(tokenizer.decode(response, skip_special_tokens=True))

            data[i]['key']={}
            data[i]['key']['entities'] = filter_entities(key_entities)
            data[i]['non_key']={}
            data[i]['non_key']['entities'] = filter_entities(non_key_entities)
            data[i]['non_key']['actions'] = filter_entities(actions)

            # for more frequent saving of annotations
            # with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)

    with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


prompt_llama_entity(dataset="vwp", portion="test")
prompt_llama_entity(dataset="sb20k", portion="test")
prompt_llama_entity(dataset="salon", portion="test")
prompt_llama_entity(dataset="vwp", portion="train")
prompt_llama_entity(dataset="sb20k", portion="train")
prompt_llama_entity(dataset="salon", portion="train")
