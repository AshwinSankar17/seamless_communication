# sleep 30m
ALIGNMENT_THESHOLD=0.8 MINING_THRESHOLD=0.6 python bhasa_anuvaad.py \
--name IndicVoices-ST \
--direction "all" \
--save_dir "data/BhasaAnuvaad" \
--hf_cache_dir "/share03/draj/TFCACHEDATA/BhasaAnuvaad" \
--huggingface_token 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 \
--do_split

# ALIGNMENT_THESHOLD=0.8 MINING_THRESHOLD=0.6 python bhasa_anuvaad.py \
# --name "Mann-ki-Baat" \
# --direction "all" \
# --save_dir "data/BhasaAnuvaad" \
# --hf_cache_dir "/share03/draj/TFCACHEDATA/BhasaAnuvaad" \
# --huggingface_token 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 \
# --do_split
