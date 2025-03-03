
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

# model_dir = "./llama31_70b_int_{vwp|sb20k|salon_short}_{cons|cons_no_attr}_lora_ft/epoch_0/"
model_dir = "./llama31_70b_int_vwp_cons_no_attr_lora_ft/epoch_0/"
# root_file = "./ft_data/{vwp|sb20k|salon_short}_test_{cons|cons_no_attr}.json"
root_file = "./ft_data/vwp_test_cons_no_attr.json"

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

with open(root_file, "r") as f:
    samples = json.load(f)

for sid, sp in tqdm(enumerate(samples)):
    messages = [
        {"role": "system", "content": "You are given a story and a target plot in the story, please generate a detailed image caption to describe the scene implied by the target plot, and generate the main characters, time and location of the scene."},
        {"role": "user", "content": sp["narrative"]},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=384,
    )
    samples[sid]["llama31_cons"] = outputs[0]["generated_text"][-1]["content"]

with open(root_file, "w") as f:
    json.dump(samples, f, indent=2)
