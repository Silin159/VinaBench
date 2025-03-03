conda activate vina_ft

cd ./mm_interleaved/models/utils/ops
python setup.py install --prefix=${HOME}
cd ../../../../

# finetune on VWP without constraints
torchrun --standalone --nproc_per_node=4 train.py --config_file=mm_interleaved/configs/release/mm_ft_vwp_no_cons.yaml --output_dir=OUTPUT/mm_interl_ft_vwp_no_cons_40k --run_name=mm_interl_ft_vwp_no_cons_40k

# finetune on VWP with constraints
torchrun --standalone --nproc_per_node=4 train.py --config_file=mm_interleaved/configs/release/mm_ft_vwp_with_cons.yaml --output_dir=OUTPUT/mm_interl_ft_vwp_with_cons_40k --run_name=mm_interl_ft_vwp_with_cons_40k

# we leave Storyboard20K for zero-shot evaluation

# finetune on StorySalon without constraints
torchrun --standalone --nproc_per_node=4 train.py --config_file=mm_interleaved/configs/release/mm_ft_storysalon_no_cons.yaml --output_dir=OUTPUT/mm_interl_ft_salon_no_cons_40k --run_name=mm_interl_ft_salon_no_cons_40k

# finetune on StorySalon with constraints
torchrun --standalone --nproc_per_node=4 train.py --config_file=mm_interleaved/configs/release/mm_ft_storysalon_with_cons.yaml --output_dir=OUTPUT/mm_interl_ft_salon_with_cons_40k --run_name=mm_interl_ft_salon_with_cons_40k
