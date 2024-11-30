TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:29400 \
    finetune.py \
    --train_dataset /data/BhasaAnuvaad/NPTEL/indic2en/assamese/train_manifest.json \
    --eval_dataset /data/BhasaAnuvaad/NPTEL/indic2en/assamese/val_manifest.json \
    --model_name seamlessM4T_v2_large \
    --save_model_to /root/repos/seamless_communication/checkpoints/pilot/dry_run.pt \
    --max_epochs 10 \
    --batch_size 8 \
    --learning_rate 2e-6 \
    --warmup_steps 1000 \
    --max_src_tokens 2500 \
    --eval_steps 1000 \
    --log_steps 100 \
    --mode SPEECH_TO_TEXT
    
    
    # /data/BhasaAnuvaad/NPTEL/en2indic/english/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/bengali/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/gujarati/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/hindi/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/kannada/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/malayalam/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/marathi/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/tamil/train_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/telugu/train_manifest.json \
    
    
    # /data/BhasaAnuvaad/NPTEL/en2indic/english/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/bengali/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/gujarati/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/hindi/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/kannada/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/malayalam/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/marathi/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/tamil/val_manifest.json \
    # /data/BhasaAnuvaad/NPTEL/indic2en/telugu/val_manifest.json \