from PIL import Image
import os
import json
import random
import numpy as np
import wget
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from .loader import BaseDataset


class VWPDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        phase="train",
        start=None,
        end=None,
        collate_mode="train",
        out_mode="images",
        add_eos="",
        num_img_token=32,
        img_first_prob=0.0,
        add_soi_token=True,
        context_type="multi_modal",
        target_image_idxs=None,
        generation_kwargs=None
    ):
        super().__init__()

        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root

        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.start = start
        self.end = end

        assert out_mode in ["images", "captions_images", "captions_links_images", "captions_setups_images", "captions_links_setups_images"]
        self.out_mode = out_mode

        assert collate_mode in ["train", "generate_images"]
        self.collate_mode = collate_mode
        self.add_eos = add_eos

        assert context_type in [
            "multi_modal",
            "image_only",
            "text_only",
            "current",
        ]
        self.context_type = context_type

        self.num_img_token = num_img_token
        self.img_first_prob = img_first_prob
        self.add_soi_token = add_soi_token

        self.image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            self.image_subseq = "<|beginofimage|>" + self.image_subseq

        self.target_image_idxs = target_image_idxs
        self.generation_kwargs = generation_kwargs
        self.save_gt_image_online = True
        self.load_database()
        print(f"length of the dataset is {len(self.annt_ids)}")

    def load_database(self):
        
        with open(self.annt_root, "r") as f:
            vwp_data = json.load(f)
            if self.start is not None and self.end is not None:
                vwp_data = vwp_data[self.start:self.end]
        
        self.annt_ids = []
        self.narratives = []
        self.captions = []
        self.images = []
        self.links = {"key_nar": [], "non_key_nar": [], "key_cap": [], "non_key_cap": []}
        self.profiles = []
        self.setups = []
        
        '''
        self.llama_captions = []
        self.cap_links = []
        self.llama_cap_links = []
        self.cap_links_setups = []
        self.llama_cap_links_setups = []
        '''

        for sample in vwp_data:
            self.annt_ids.append(sample["scene_full_id"]+"_"+str(sample["story_id"]))
            self.narratives.append(sample["narrative"])
            self.images.append(sample["image_links"])
            self.profiles.append(sample["profile"])
            self.setups.append(sample["setups_no_desp"])

            # if "captions" in self.out_mode:
            self.captions.append(sample["captions"])
            
            '''
            if self.phase == "test":
                self.llama_captions.append(sample["llama31_caps"])
                self.cap_links.append(sample["captions_links"])
                self.llama_cap_links.append(sample["llama31_cap_links"])
                self.cap_links_setups.append(sample["captions_links_setups"])
                self.llama_cap_links_setups.append(sample["llama31_cap_links_setups"])
            '''
            
            # if "links" in self.out_mode:
            link_map = {"nar": "links_to_nar", "cap": "links_between_cap"}
            for ent_type in ["key", "non_key"]:
                for link_type in ["nar", "cap"]:
                    links_set = []
                    if link_type == "cap":
                        links_set.append({})
                    for raw_link in sample[ent_type][link_map[link_type]]:
                        links = {}
                        for match in re.findall(r"\([^,\(\)]+, [^,\(\)]+\)", raw_link):
                            entity1 = match.split(", ")[0].strip("( )")
                            entity2 = match.split(", ")[1].strip("( )")
                            if link_type == "nar":
                                links[entity1] = entity2  # caption --> narrative
                            else:
                                links[entity2] = entity1  # current --> previous caption
                        links_set.append(links)
                    self.links[ent_type+"_"+link_type].append(links_set)

        # print(self.annt_ids[0])
        # print([self.links[k][0] for k in self.links])

    def __repr__(self) -> str:
        return (
            f"VWP Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annt_ids)

    def _get_narrative(self, sid):
        return self.narratives[sid]
    
    def _get_caption(self, sid, add_links=False):
        # TO DO: add links
        if add_links:
            key_nar_links = self.links["key_nar"][sid]
            captions = self.captions[sid]
            
            captions_with_links = []
            for cap, link_set in zip(captions, key_nar_links):
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
                
                captions_with_links.append(" ".join(cap_ls))
            return captions_with_links
        else:
            return self.captions[sid]

    def _get_img_links(self, sid):
        return self.images[sid]
    
    def _get_global_id(self, sid):
        return self.annt_ids[sid]
    
    def _get_setups(self, sid):
        return self.setups[sid]
    
    def _get_profile(self, sid):
        return self.profiles[sid]
    
    def _get_image(self, img_link, transfrom=True):
        out_pth = os.path.join(self.data_root, img_link.split("/")[-2])
        img_file = os.path.join(out_pth, img_link.split("/")[-1])
        os.makedirs(out_pth, exist_ok=True)
        if not os.path.exists(img_file):
            wget.download(img_link, out=out_pth)
        
        # padding to square
        '''
        img_ori = Image.open(img_file).convert("RGB")
        W, H = img_ori.size
        pad_size = max(W, H)
        pad_w = (pad_size - W) // 2
        pad_h = (pad_size - H) // 2
        img = Image.new(img_ori.mode, (pad_size, pad_size), (0, 0, 0))
        img.paste(img_ori, (pad_w, pad_h))
        '''
        img = Image.open(img_file).convert("RGB")
        if transfrom:
            img_arr_tuple = self.transform(img)  # (in_img_arr, out_img_arr)
            return img_arr_tuple
        else:
            return img

    def meta_to_image(self, meta, target_image_idx=-1):
        img_link = self.images[int(meta[1])][target_image_idx]
        image = self._get_image(img_link, transfrom=False)
        return image

    def __getitem__(self, index):
        
        global_id = self._get_global_id(index)
        meta = [global_id, str(index)]
        full_narrative = self._get_narrative(index)
        if "captions" in self.out_mode:
            if "links" in self.out_mode:
                image_captions = self._get_caption(index, add_links=True)
            else:
                image_captions = self._get_caption(index, add_links=False)
            if "setups" in self.out_mode:
                profile = self._get_profile(index)
                setups = self._get_setups(index)
        image_links = self._get_img_links(index)
        
        images_tensor = []
        text = ""

        if self.collate_mode == "train":
            assert self.phase == "train"
        else:
            assert self.phase != "train"
            assert self.context_type == "multi_modal"

        if "setups" in self.out_mode:
            text += f"Character Profile: "
            if len(profile) > 0:
                for char, desp in profile.items():
                    text += f"{char} -- {desp}; "
                assert text[-2] == ";"
                text = text[:-2]
                text += f". "
            else:
                text += f"(none). "
        
        for i in range(len(image_links)):
            sub_text = ""
            image_tuple = self._get_image(image_links[i])
            images_tensor.append(image_tuple)

            narrative = full_narrative[i]
            sub_text += f"Plot {str(i)}: {narrative} "
            
            if "captions" in self.out_mode:
                caption = image_captions[i]
                # may need early truncate
                sub_text += f"Caption {str(i)}: {caption} "
                
                sub_text_token = sub_text.split(" ")
                if len(sub_text_token) > 100:
                    sub_text_token = sub_text_token[:100]
                text += " ".join(sub_text_token)
            
                if "setups" in self.out_mode:
                    setup = setups[i]
                    # setup = setup.replace("Characters: ", "[Characters] ")
                    # setup = setup.replace("\nTime: ", " [Time] ")
                    # setup = setup.replace("\nLocation: ", " [Location] ")
                    text += f" {setup} "
            else:
                text += sub_text
            
            text += f"Image {str(i)}: {self.image_subseq} "

        # add padding image for cold-start MMFS
        # text = f"Blank Image: {self.image_subseq} " + text
        # pad_in = np.zeros_like(images_tensor[0][0])
        # pad_out = np.zeros_like(images_tensor[0][1])
        # images_tensor = [(pad_in, pad_out)] + images_tensor
        
        text = text.strip()
        if self.add_eos:
            text += self.add_eos

        return dict(text=text, images_tensor=images_tensor, meta=meta)

    @property
    def task_prefix(self):
        return f"_{self.context_type}"
