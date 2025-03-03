import os
import os.path as osp
import json
from PIL import Image
from datetime import datetime
import numpy as np
import torch
from torchvision.utils import save_image
import sys
sys.path.append("/home/sigao/local/lib/python3.10/dist-packages/MultiScaleDeformableAttention-1.0-py3.10-linux-x86_64.egg")

from mm_interleaved.models.utils.monkey_patch import (
    replace_llama_attn_with_flash_attn,
    replace_blip2_attn_with_qknorm_attn,
    replace_beam_search,
    replace_stable_diffusion_pipeline_call,
    replace_stable_diffusion_unet_forward,
)
replace_beam_search()
replace_blip2_attn_with_qknorm_attn()
replace_stable_diffusion_unet_forward()
replace_stable_diffusion_pipeline_call()
IS_TRAIN = False
if IS_TRAIN:
    replace_llama_attn_with_flash_attn()


from mm_interleaved.models import MMInterleaved
from mm_interleaved.custom_datasets.utils import create_transform
from mm_interleaved.custom_datasets.wds_utils import init_tokenizer
from mm_interleaved.utils import (
    ArgumentParser,
    TrainingArguments,
    init_distributed_mode,
    load_model_weights,
)
from mm_interleaved.utils.clip_sim_score import tensor_to_pil, calculate_clip_sim_i2i
from mm_interleaved.utils.fid_score import calculate_fid_given_paths


def get_image(data_root, img_pth, transform=None):
    img_file = os.path.join(data_root, img_pth)
    img = Image.open(img_file).convert("RGB")
    if transform is not None:
        img_arr_in, img_gold = transform(img)
        return img_arr_in, img_gold
    else:
        return img


def load_annt_data(
    transform,
    tokenizer,
    num_total_token=2048,
    truncation=True,
    num_img_token=64,
    generation_kwargs=None,
    data_root="",
    annt_path="",
    start=None,
    end=None,
    out_dir="",
):
    with open(annt_path, "r") as rf:
        infos = json.load(rf)
        infos = infos[start:end]

    data = []
    for info in infos:
        images = []
        text = ""
        image_subseq = "<|beginofimage|>" + "<|image|>" * num_img_token
        
        start_idx = info.get("start_idx", 0)       
        if start_idx == 0:
            info["start_idx"] = 0
            
            # add padding image
            W = transform.transform1.resolution
            H = int(transform.transform1.hw_ratio * W)
            images.append(np.zeros((3, H, W), dtype=np.float32))
            
            narrative = info["narrative"][start_idx]
            text += f"Plot {str(start_idx)}: {narrative} "

            text += f"Image {str(start_idx)}: {image_subseq} "
        
        else:
            for idx in range(start_idx):
                scene_id = info["movie_id"]
                story_id = str(info["global_id"])
                
                img_in, img_gold = get_image(data_root, info["key_frames"][idx], transform)
                
                gold_img_path = out_dir+"/gold/"+scene_id+"_"+story_id+"_"+str(idx)+".jpg"
                img_gold.save(gold_img_path)
                
                images.append(img_in)

                narrative = info["narrative"][idx]
                text += f"Plot {str(idx)}: {narrative} "
                text += f"Image {str(idx)}: {image_subseq} "

        assert len(images) > 0, "Please provide at least 1 image as inputs"
        image_tensors = np.stack(images, axis=0)
        
        text = text.strip()
        tokenizer.padding_side = "right"
        text_tensor = tokenizer(
            text,
            max_length=num_total_token,
            truncation=truncation,
            padding=False,
            return_tensors="np",
            return_attention_mask=True,
        )
        text_ids = text_tensor["input_ids"]
        text_attn_mask = text_tensor["attention_mask"]

        image_tensors = torch.from_numpy(image_tensors)
        num_images = image_tensors.shape[0]
        target_image_idxs = torch.tensor([num_images - 1], dtype=torch.long)

        _data = dict(
            image_tensors=image_tensors,
            image_tensors_dec=None,
            text_ids=torch.from_numpy(text_ids),
            attention_mask=torch.from_numpy(text_attn_mask),
            num_image_per_seq=torch.tensor([num_images]),
            nearest_bos_idxs=None,
            meta=info,
            target_image_idxs=target_image_idxs
        )
        assert len(_data["text_ids"].shape) == 2 and _data["text_ids"].shape[0] == 1

        if generation_kwargs is not None:
            for k, v in generation_kwargs.items():
                _data[k] = v

        data.append(_data)

    return data


def update_texts(
    inputs,
    new_text,
    num_img_token=64,
    pad_img_tensor=None,
    tokenizer=None
):
    
    assert tokenizer is not None
    soi_token_id = tokenizer.convert_tokens_to_ids("<|beginofimage|>")
    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    
    new_text_tensor = tokenizer(new_text, max_length=200, truncation=True,
                padding=False, return_tensors="pt", return_attention_mask=True)
    new_text_ids = new_text_tensor["input_ids"]
    
    image_ids = [image_token_id] * num_img_token
    image_ids = [soi_token_id] + image_ids
    image_ids = torch.tensor(image_ids).type_as(new_text_ids)
    image_ids = image_ids.unsqueeze(0)

    text_ids = inputs["text_ids"]
    new_ids = torch.cat((new_text_ids, image_ids), dim=-1)
    new_ids = new_ids.to(device="cuda")

    attention_mask = inputs["attention_mask"]
    new_attn_mask = torch.ones_like(new_ids)
    new_attn_mask = new_attn_mask.to(device="cuda")
    
    inputs["text_ids"] = torch.cat((text_ids, new_ids), dim=-1)
    inputs["text_ids"] = inputs["text_ids"].to(device="cuda")
    inputs["attention_mask"] = torch.cat((attention_mask, new_attn_mask), dim=-1)
    inputs["attention_mask"] = inputs["attention_mask"].to(device="cuda")

    image_tensors = inputs["image_tensors"]
    pad_img_tensor =  pad_img_tensor.to(device="cuda")
    inputs["image_tensors"] = torch.cat((image_tensors, pad_img_tensor), dim=0)
    inputs["image_tensors"] = inputs["image_tensors"].to(device="cuda")
    
    inputs["target_image_idxs"] = inputs["target_image_idxs"] + 1
    inputs["target_image_idxs"] = inputs["target_image_idxs"].to(device="cuda")
    inputs["num_image_per_seq"] = inputs["num_image_per_seq"] + 1
    inputs["num_image_per_seq"] = inputs["num_image_per_seq"].to(device="cuda")


def update_image(inputs, images, transform=None):
    assert len(images) == 1
    pil_images = tensor_to_pil(images)
    image_tensor_pred, _ = transform(pil_images[0])
    if isinstance(image_tensor_pred, np.ndarray):
        image_tensor_pred = torch.from_numpy(image_tensor_pred)
    # update: image_tensors
    image_tensor_pred = image_tensor_pred.to(device="cuda")
    inputs["image_tensors"][-1, ...] = image_tensor_pred
    inputs["image_tensors"] = inputs["image_tensors"].to(device="cuda")


def inference_all(model, config, data_root, annt_path, start, end, output_dir):
    # prepare data
    tokenizer = init_tokenizer(config.tokenizer_path)
    transform = create_transform(**config.transform)

    gold_save_dir = output_dir+"/gold"
    pred_save_dir = output_dir+"/pred"
    os.makedirs(gold_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)

    data = load_annt_data(
        transform=transform,
        tokenizer=tokenizer,
        num_img_token=config.num_img_token,
        generation_kwargs=config.generation_kwargs,
        data_root=data_root,
        annt_path=annt_path,
        start=start,
        end=end,
        out_dir=output_dir
    )

    W_in = transform.transform1.resolution
    H_in = int(transform.transform1.hw_ratio * W_in)
    pad_img = torch.zeros((1, 3, H_in, W_in))

    W_out = transform.transform2.resolution
    H_out = int(transform.transform2.hw_ratio * W_out)

    eval_image_pairs = []
    eval_image_id = 0
    # suffix = datetime.now().strftime("%Y%m%d%H%M")
    print("Inference Start")
    for sample_idx, inputs in enumerate(data):
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device="cuda")
                inputs[k] = v
        
        meta = inputs["meta"]
        sid = meta["movie_id"]+"_"+str(meta["global_id"])
        
        narrative = meta["narrative"]
        start_idx = meta["start_idx"]
        with torch.no_grad():
            for idx in range(start_idx, len(narrative)):
                
                if idx > 0:
                    plot = narrative[idx]
                    new_text = f"Plot {str(idx)}: {plot} "
                    new_text += f"Image {str(idx)}:"
                    update_texts(inputs, new_text, pad_img_tensor=pad_img, tokenizer=tokenizer)
                else:
                    pass

                inputs["gen_h"] = H_out
                inputs["gen_w"] = W_out

                outputs = model.generate(mode="generate_images", **inputs)

                update_image(inputs, outputs["image"][0].unsqueeze(0), transform=transform)
                
                pred_img_pth = os.path.join(pred_save_dir, sid+"_"+str(idx)+".jpg")
                save_image(outputs["image"][0], pred_img_pth)

                _, gold_img = get_image(data_root, meta["key_frames"][idx], transform)
                gold_img_pth = os.path.join(gold_save_dir, sid+"_"+str(idx)+".jpg")
                gold_img.save(gold_img_pth)

                img_result = {"story_id": sid, "sample_idx": idx, "image_id": eval_image_id,
                              "image_path": pred_img_pth, "image_gt_path": gold_img_pth}
                eval_image_pairs.append(img_result)
                eval_image_id += 1

    print("All finished")


def main():
    parser = ArgumentParser(TrainingArguments)
    init_distributed_mode()
    args = parser.parse_args_with_config_file_into_dataclasses()
    train_args, config = args
    print(train_args)
    print(config)

    print("Model Init Start")
    model = MMInterleaved(hw_ratio=config.inference.transform.hw_ratio, **config.model)
    image_size = config.inference.transform.resolution
    model.visual_tokenizer.encoder.interpolate_pos_embed(image_size, hw_ratio=model.hw_ratio)

    if getattr(config, "load_from", None):
        load_model_weights(model, config.load_from)
    model = model.to(device="cuda")
    model.eval()

    inference_all(model=model, config=config.inference, data_root=config.data_root, annt_path=config.annt_path,
                  start=config.start, end=config.end, output_dir=train_args.output_dir)


if __name__ == "__main__":
    main()
