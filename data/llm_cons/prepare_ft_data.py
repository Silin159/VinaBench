import json
import re
from nltk.tokenize import word_tokenize
from copy import deepcopy

root_file = "../annotations/$_*.json"
out_file_cons = "./ft_data/$_*_cons.json"  # for ARLDM and StoryGen
out_file_cons_no_attr = "./ft_data/$_*_cons_no_attr.json"  # for MM-Interleaved

for dataset in ["vwp", "sb20k", "salon_short"]:  # "sb20k" is not used in our expriments
    for split in ["train", "test"]:
        with open(root_file.replace("$", dataset).replace("*", split), "r") as f:
            data = json.load(f)
        
        output_data_cons = []
        output_data_cons_no_attr = []
        for sample in data:
            full_narrative = sample["narrative"]
            full_cons = sample["captions_links_setups"]
            full_cons_no_attr = sample["captions_links_setups_no_desp"]
            
            input_text_base = "Story:\n" + "\n".join(full_narrative) + "\n\n"
            for tid, plot in enumerate(full_narrative):
                input_text = deepcopy(input_text_base) + "Target Plot:\n" + plot
                output_data_cons.append({"narrative": input_text, "gold_cons": full_cons[tid]})
                output_data_cons_no_attr.append({"narrative": input_text, "gold_cons": full_cons_no_attr[tid]})

        with open(out_file_cons.replace("$", dataset).replace("*", split), "w") as f:
            json.dump(output_data_cons, f, indent=2)
        with open(out_file_cons_no_attr.replace("$", dataset).replace("*", split), "w") as f:
            json.dump(output_data_cons_no_attr, f, indent=2)
