# FID and CLIP-I scores
cd ../MM-Interleaved
conda activate vina_ft
python calculate_fid_clipi.py

# CLIP-T and CLIP-T-MRR scores
cd ../evaluation
python prepare_mrr_rank_clip.py
python clip_text_rank.py

# LLaVA-VQA-MRR scores (slow)
cd ../LLaVA-NeXT
conda activate vina_llava_onev
# python prepare_mrr_rank_full_llava.py  # select candidates from all images, very slow
python prepare_mrr_rank_clip_llava.py  # alternative, re-rank top-100 candidates selected by CLIP
python vqa_text_rank.py

# LLaVA-VQA-Alignment and Consistency scores (slow)
# Non-character entity alignment
python vqa_entity_align.py
# Character (number and attributes) alignment
python vqa_character_align.py
# Time alignment
python vqa_time_align.py
# Location alignment
python vqa_location_align.py
# Style consistency
python vqa_style_consist.py
# Character consistency
python vqa_character_consist.py
# Location consistency
python vqa_location_consist.py

# MiniCPM-VQA-Alignment and Consistency scores
cd ../evaluation
conda activate vina_minicpm
# Non-character entity alignment
python minicpm_vqa_entity_align.py
# Character (number and attributes) alignment
python minicpm_vqa_character_align.py
# Time alignment
python minicpm_vqa_time_align.py
# Location alignment
python minicpm_vqa_location_align.py
# Style consistency
python minicpm_vqa_style_consist.py
# Character consistency
python minicpm_vqa_character_consist.py
# Location consistency
python minicpm_vqa_location_consist.py
