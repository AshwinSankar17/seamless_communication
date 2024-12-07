ALIGNMENT_THESHOLD=0.8 MINING_THRESHOLD=0.6 python bhasa_anuvaad.py \
--name "all" \
--direction "all" \
--save_dir "/data/BhasaAnuvaad" \
--hf_cache_dir "/data/huggingface/datasets/BhasaAnuvaad" \
--huggingface_token $HUGGINGFACE_TOKEN \
--do_split