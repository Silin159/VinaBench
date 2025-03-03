import json
import re
from nltk.tokenize import word_tokenize
from copy import deepcopy


def extract_entities(raw_link, plot):
    entities = []
    for match in re.findall(r"\([^,\(\)]+, [^,\(\)]+\)", raw_link):
        entity = match.split(", ")[1].strip("( )")
        if entity in plot:
            entities.append(entity)
    return entities


def create_constraints(dataset="vwp", portion="test"):
    '''
    dataset: ["vwp", "sb20k", "salon"]
    portion: ["train", "test"]
    '''
    print(f"Aggregating annotations into constraints for {dataset} {portion} set!")

    with open("../annotations/"+dataset+"_"+portion+".json", "r") as f:
        data = json.load(f)

    filtered_data = []
    for i in range(len(data)):
        setups = []
        setups_no_desp = []
        for tid in range(len(data[i]["narrative"])):
            chars = []
            chars_no_desp = []
            for char in data[i]["scene_characters"][tid]["present"]:
                if char in data[i]["global_profile"]:
                    chars.append(char + "(" + data[i]["global_profile"][char] + ")")
                    chars_no_desp.append(char)
                
            if len(chars) > 0:
                setup = "[Characters] " + ", ".join(chars)
            else:
                setup = "[Characters] (none)"
            setup += " [Time] " + data[i]["time"][tid]
            setup += " [Location] " + data[i]["location"][tid]
            setups.append(setup)

            if len(chars_no_desp) > 0:
                setup_no_desp = "[Characters] " + ", ".join(chars_no_desp)
            else:
                setup_no_desp = "[Characters] (none)"
            setup_no_desp += " [Time] " + data[i]["time"][tid]
            setup_no_desp += " [Location] " + data[i]["location"][tid]
            setups_no_desp.append(setup_no_desp)

        data[i]["setups"] = setups
        data[i]["setups_no_desp"] = setups_no_desp

        key_links = []
        for link in data[i]["key"]["links_to_nar"]:
            lk = {}
            for match in re.findall(r"\([^,\(\)]+, [^,\(\)]+\)", link):
                entity1 = match.split(", ")[0].strip("( )")
                entity2 = match.split(", ")[1].strip("( )")
                lk[entity1] = entity2
            key_links.append(lk)
            
        captions_links = []
        for cap, link_set in zip(data[i]["captions"], key_links):
            idx = 0
            cap_ls = word_tokenize(cap)

            for cap_ent, nar_ent in link_set.items():
                cap_ent_ls = word_tokenize(cap_ent)
                len_ce = len(cap_ent_ls)
                nar_ent_ls = ["("] + word_tokenize(nar_ent)+ [")"]

                while idx < len(cap_ls)-len_ce+1:
                    if cap_ls[idx:idx+len_ce] == cap_ent_ls:
                        break
                    idx += 1
                    
                if idx < len(cap_ls)-len_ce+1:
                    cap_ls = cap_ls[:idx+len_ce] + nar_ent_ls + cap_ls[idx+len_ce:]
                    idx = idx + len_ce + len(nar_ent_ls)
                
            captions_links.append(" ".join(cap_ls))
            
        data[i]["captions_links"] = captions_links

        captions_setups = []
        captions_setups_no_desp = []
        for tid, cap in enumerate(data[i]["captions"]):
            captions_setups.append(cap + " " + data[i]["setups"][tid])
            captions_setups_no_desp.append(cap + " " + data[i]["setups_no_desp"][tid])
        data[i]["captions_setups"] = captions_setups
        data[i]["captions_setups_no_desp"] = captions_setups_no_desp
            
        captions_links_setups = []
        captions_links_setups_no_desp = []
        for tid, cap_link in enumerate(data[i]["captions_links"]):
            captions_links_setups.append(cap_link + " " + data[i]["setups"][tid])
            captions_links_setups_no_desp.append(cap_link + " " + data[i]["setups_no_desp"][tid])
            
        data[i]["captions_links_setups"] = captions_links_setups
        data[i]["captions_links_setups_no_desp"] = captions_links_setups_no_desp

        key_entities_count = 0
        raw_link_key = data[i]["key"]["links_to_nar"]
        for tid, plot in enumerate(data[i]["narrative"]):
            nar_ent_key = extract_entities(raw_link_key[tid], plot)
            key_entities_count += len(set(nar_ent_key))
        
        if key_entities_count == 0 or len(data[i]["global_profile"]) == 0:
            pass
        else:
            filtered_data.append(data[i])

    print(len(data))
    print(len(filtered_data))

    with open("../annotations/"+dataset+"_"+portion+".json", "w") as f:
        json.dump(filtered_data, f, indent=2)


create_constraints(dataset="vwp", portion="test")
create_constraints(dataset="sb20k", portion="test")
create_constraints(dataset="salon", portion="test")
create_constraints(dataset="vwp", portion="train")
create_constraints(dataset="sb20k", portion="train")
create_constraints(dataset="salon", portion="train")
