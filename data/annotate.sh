cd scripts
conda activate vina_mantis
# image captioning
python image_caption_mantis.py

conda activate vina_llama
# entity extraction
python entity_extraction_llama3.py
# entity (commonsense) linking
python entity_linking_llama3.py
# global character list parsing
python global_char_llama3.py
# global character attribute parsing
python global_attr_llama3.py

cd ../../LLaVA-NeXT
conda activate vina_llava_onev
# scene-specific presented character parsing
python scene_char_llava.py
# scene-specific time and location parsing
python scene_time_loc_llava.py

cd ../data/scripts
conda activate vina_minicpm
# global image style parsing
python global_style_minicpm.py

# create gold constraints
python create_gold_constraints.py

# split long storyboards in StorySalon to facilitate baseline model training
python salon_scene_split.py

# prepare data to finetune LLM for generating narrative constraints
cd ../llm_cons
python prepare_ft_data.py

# finetune LLM (Llama-3.1-70B-Instruct) with LoRA
conda activate vina_llm_cons
# download Llama-3.1-70B-Instruct (need HuggingFace token login)
tune download meta-llama/Llama-3.1-70B-Instruct --output-dir ./Llama-3.1-70B-Instruct --ignore-patterns "original/consolidated*" --hf-token "***"
# please set the input finetuning data and output directory in .yaml file
tune run --nproc_per_node 8 lora_finetune_distributed --config ./llama_70B_lora_ft.yaml

# generate narrative constraints
python run_inf_cons.py

# write llm-generated constraints into annotations
python write_llm_cons.py
