import json

cut_map = {11: [6, 11], 12: [6, 12], 13: [7, 13], 14: [7, 14], 15: [8, 15],
           16: [8, 16], 17: [9, 17], 18: [9, 18], 19: [10, 19], 20: [10, 20],
           21: [7, 14, 21], 22: [8, 15, 22], 23: [8, 16, 23], 24: [8, 16, 24],
           25: [9, 17, 25], 26: [9, 18, 26], 27: [9, 18, 27], 28: [10, 19, 28],
           29: [10, 20, 29], 30: [10, 20, 30]}

'''further split for ARLDM baseline training
cut_map = {6: [3, 6], 7: [4, 7], 8: [4, 8], 9: [5, 9], 10: [5, 10]}
'''

for split in ["train", "test"]:
    cut_sample = []
    with open("../annotations/salon_"split+".json", "r") as f:
        samples = json.load(f)
    
    for sp in samples:
        sp_len = len(sp["image_paths"])
        if sp_len in cut_map:
            prev_cut = 0
            for cid, cut in enumerate(cut_map[sp_len]):
                sp_cut = {}
                sp_cut["portion"] = sp["portion"]
                sp_cut["sid"] = sp["sid"] + "_" + str(cid)
                sp_cut["global_profile"] = sp["global_profile"]
                sp_cut["style"] = sp["style"]
                for key in ["image_paths", "narrative", "captions",
                            "key:entities", "key:links_to_nar", "non_key:entities",
                            "non_key:links_to_nar", "non_key:actions", "linked_entities",
                            "scene_characters", "time", "location", "setups",
                            "setups_no_desp", "captions_links", "captions_setups",
                            "captions_setups_no_desp", "captions_links_setups",
                            "captions_links_setups_no_desp"]:
                    if ":" in key:
                        key1 = key.split(":")[0]
                        if key1 not in sp_cut:
                            sp_cut[key1] = {}
                        key2 = key.split(":")[1]
                        sp_cut[key1][key2] = sp[key1][key2][prev_cut:cut]
                    else:
                        sp_cut[key] = sp[key][prev_cut:cut]
                cut_sample.append(sp_cut)
                prev_cut = cut
        else:
            cut_sample.append(sp)
    
    print(len(cut_sample))
    with open("../annotations/salon_short_"split+".json", "w") as f:
        json.dump(cut_sample, f, indent=2)
