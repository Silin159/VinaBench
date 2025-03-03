import json

annot_file = "../annotations/$_test.json"
llm_cons_file = "./ft_data/$_test_cons.json"
llm_cons_no_attr_file = "./ft_data/$_test_cons_no_attr.json"

for dataset in ["vwp", "sb20k", "salon_short"]:
    with open(annot_file.replace("$", dataset), "r") as f:
        annot_data = json.load(f)
    with open(llm_cons_file.replace("$", dataset), "r") as f:
        llm_cons_data = json.load(f)
    with open(llm_cons_no_attr_file.replace("$", dataset), "r") as f:
        llm_cons_no_attr_data = json.load(f)

    index = 0
    for sid, sp in enumerate(annot_data):
        annot_data[sid]["llama31_cap_links_setups"] = []
        annot_data[sid]["llama31_cap_links_setups_no_desp"] = []
        
        for tid in range(len(sp["narrative"])):
            annot_data[sid]["llama31_cap_links_setups"].append(llm_cons_data[index]["llama31_cons"])
            annot_data[sid]["llama31_cap_links_setups_no_desp"].append(llm_cons_no_attr_data[index]["llama31_cons"])

            index += 1

    with open(annot_file.replace("$", dataset), "w") as f:
        json.dump(annot_data, f, indent=2)
