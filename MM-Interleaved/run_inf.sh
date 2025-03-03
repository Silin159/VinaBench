conda activate vina_ft

cd ./mm_interleaved/models/utils/ops
python setup.py install --prefix=${HOME}
cd ../../../../

# testing on VWP without constraints
torchrun inference_no_cons_vwp.py --config_file=mm_interleaved/configs/release/mm_inf_vwp_no_cons.yaml

# testing on VWP with LLM-generated constraints
torchrun inference_llama_cons_vwp.py --config_file=mm_interleaved/configs/release/mm_inf_vwp_llama_cons.yaml

# testing on VWP with gold constraints
torchrun inference_gold_cons_vwp.py --config_file=mm_interleaved/configs/release/mm_inf_vwp_gold_cons.yaml

# testing on Storyboard20K without constraints
torchrun inference_no_cons_sb20k.py --config_file=mm_interleaved/configs/release/mm_inf_sb20k_no_cons.yaml

# testing on Storyboard20K with LLM-generated constraints
torchrun inference_llama_cons_sb20k.py --config_file=mm_interleaved/configs/release/mm_inf_sb20k_llama_cons.yaml

# testing on Storyboard20K with gold constraints
torchrun inference_gold_cons_sb20k.py --config_file=mm_interleaved/configs/release/mm_inf_sb20k_gold_cons.yaml

# testing on StorySalon without constraints
torchrun inference_no_cons_salon.py --config_file=mm_interleaved/configs/release/mm_inf_salon_no_cons.yaml

# testing on StorySalon with LLM-generated constraints
torchrun inference_llama_cons_salon.py --config_file=mm_interleaved/configs/release/mm_inf_salon_llama_cons.yaml

# testing on StorySalon with gold constraints
torchrun inference_gold_cons_salon.py --config_file=mm_interleaved/configs/release/mm_inf_salon_gold_cons.yaml
