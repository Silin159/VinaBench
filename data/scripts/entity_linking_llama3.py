from copy import deepcopy
import json
import os
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from transformers.image_utils import load_image
import sys
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

base_messages_nar_link_key = [
    {"role": "system", "content": "You are given a caption, a narrative statement, and an entity in the caption. \
        If there is a link between the caption entity and an entity in the narrative, output the link. \
        If there is no link for a caption entity, report 'no link'. Do not give any explanation in your answer."},
    {
        "role": "user",
        "content": "Caption: A woman with a sad face is sitting at the table, opposite her is another woman reading a newspaper. \
    Narrative: They chose a table to sit down, while Elle told Karen a piece of bad news on the newspaper. \
    Caption entity: woman with a sad face"
    },
    {
        "role": "assistant",
        "content": "(woman with a sad face, Karen)"
    },
    {
        "role": "user",
        "content": "Caption: The reddish orange sun is slightly visible at the horizon as it rises. The sky is mixed with pink and orange clouds. \
    The ocean waves are crashing against the sand of the beach. Three people run towards the water, each holding a surfboard. A lifeguard sits \
    near the edge of the water. \
    Narrative: The three friends went to the beach at dawn to surf. \
    Caption entity: people"
    },
    {
        "role": "assistant",
        "content": "(people, friends)"
    },
    {
        "role": "user",
        "content": "Caption: The reddish orange sun is slightly visible at the horizon as it rises. The sky is mixed with pink and orange clouds. \
    The ocean waves are crashing against the sand of the beach. Three people run towards the water, each holding a surfboard. A lifeguard sits \
    near the edge of the water. \
    Narrative: The three friends went to the beach at dawn to surf. \
    Caption entity: lifeguard"
    },
    {
        "role": "assistant",
        "content": "no link"
    }
]

base_messages_nar_link_non_key = [
    {"role": "system", "content": "You are given a caption, a narrative statement, and an entity in the caption. \
        If there is a link between the caption entity and an entity in the narrative, output the link. \
        If there is no link for a caption entity, report 'no link'. Do not give any explanation in your answer."},
    {
        "role": "user",
        "content": "Caption: A woman with a sad face is sitting at the table, opposite her is another woman reading a newspaper. \
    Narrative: They chose a table to sit down, while Elle told Karen a piece of bad news on the newspaper. \
    Caption entity: newspaper"
    },
    {
        "role": "assistant",
        "content": "(newspaper, newspaper)"
    },
    {
        "role": "user",
        "content": "Caption: The reddish orange sun is slightly visible at the horizon as it rises. The sky is mixed with pink and orange clouds. \
    The ocean waves are crashing against the sand of the beach. Three people run towards the water, each holding a surfboard. \
    Narrative: The three friends went to the beach at dawn to surf. \
    Caption entity: surfboard"
    },
    {
        "role": "assistant",
        "content": "(surfboard, surf)"
    },
    {
        "role": "user",
        "content": "Caption: The reddish orange sun is slightly visible at the horizon as it rises. The sky is mixed with pink and orange clouds. \
    The ocean waves are crashing against the sand of the beach. Three people run towards the water, each holding a surfboard. \
    Narrative: The three friends went to the beach at dawn to surf. \
    Caption entity: clouds"
    },
    {
        "role": "assistant",
        "content": "no link"
    }
]


def cleaning_links(raw_links):
    clean_links = []
    for raw_lk in raw_links:
        clean_lk = []
        for match in re.findall(r"\([^,\(\)]+, [^,\(\)]+\)", raw_lk):
            clean_lk.append(match)
        clean_links.append(" ".join(clean_lk))
    return clean_links


def prompt_with_msg(base_messages, new_messages):
    messages = base_messages + [{"role": "user", "content": new_messages}]
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
    return tokenizer.decode(response, skip_special_tokens=True)


def cap_nar_linking_key(narrative_list, caption_list, key_entities):
    nar_key_links = []
    narrative_i = ""

    for idx, entity_str in enumerate(key_entities):
        
        narrative_i += " " + narrative_list[idx]
        narrative_i = narrative_i.strip()
        nar_links_list = ""
        
        if entity_str != "none":
            entity_list = entity_str.split(", ")
        else:
            entity_list = []
        
        for entity in entity_list:
            messages = f"Caption: {caption_list[idx]} Narrative: {narrative_i} Caption entity: {entity}"
            nar_links_list += " " + prompt_with_msg(base_messages_nar_link_key, messages)

        nar_key_links.append(nar_links_list.strip())

    return cleaning_links(nar_key_links)


def cap_nar_linking_non_key(narrative_list, caption_list, key_entities):
    nar_non_key_links = []
    narrative_i = ""

    for idx, entity_str in enumerate(key_entities):
        
        narrative_i += " " + narrative_list[idx]
        narrative_i = narrative_i.strip()
        nar_links_list = ""
        
        if entity_str != "none":
            entity_list = entity_str.split(", ")
        else:
            entity_list = []
        
        for entity in entity_list:
            messages = f"Caption: {caption_list[idx]} Narrative: {narrative_i} Caption entity: {entity}"
            nar_links_list += " " + prompt_with_msg(base_messages_nar_link_non_key, messages)

        nar_non_key_links.append(nar_links_list.strip())

    return cleaning_links(nar_non_key_links)


def linked_entities(link, plot):
    entities = []
    for match in re.findall(r"\([^,\(\)]+, [^,\(\)]+\)", link):
        entity = match.split(", ")[1].strip("( )")
        if entity in plot:
            entities.append(entity)
    return entities


def end_to_end_linking(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Annotating commonsense links of {dataset} {portion} set!")

    with open("../annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)

    for i in range(len(data)):

        if 'links_to_nar' not in data[i]['key'] and len(data[i]["narrative"]) <= 30:
            
            narratives = data[i]["narrative"]
            captions = data[i]["captions"]
            key_entities = data[i]['key']['entities']
            non_key_entities = data[i]['non_key']['entities']

            nar_key_links = cap_nar_linking_key(narratives, captions, key_entities)
            nar_non_key_links = cap_nar_linking_non_key(narratives, captions, non_key_entities)

            data[i]['key']['links_to_nar'] = nar_key_links
            data[i]['non_key']['links_to_nar'] = nar_non_key_links

            entities = []
            for tid, plot in enumerate(narratives):
                nar_ent = linked_entities(nar_key_links[tid], plot) + linked_entities(nar_non_key_links[tid], plot)
                nar_ent = list(set(nar_ent))
                entities.append("; ".join(nar_ent))
            data[i]["linked_entities"] = entities

            # for more frequent saving of annotations
            # with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
            #     json.dump(data, f, indent=2)
    
    with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(data, f, indent=2)


end_to_end_linking(dataset="vwp", portion="test")
end_to_end_linking(dataset="sb20k", portion="test")
end_to_end_linking(dataset="salon", portion="test")
end_to_end_linking(dataset="vwp", portion="train")
end_to_end_linking(dataset="sb20k", portion="train")
end_to_end_linking(dataset="salon", portion="train")
