# To start multi-node training, change nnodes to $NUM_NODES and rdzv-endpoint to $MASTER_ADDR:$MASTER_PORT in all nodes
export CUDA_VISIBLE_DEVICES=0

OMP_NUM_THREADS=16 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:8009 \
    --no-python \
    m4t_finetune \
    --train_dataset data/BhasaAnuvaad/Mann-ki-Baat/indic2en/punjabi/train_manifest.json \
    --eval_dataset data/BhasaAnuvaad/Mann-ki-Baat/indic2en/punjabi/test_manifest.json \
    --model_name seamlessM4T_v2_large \
    --save_model_to models/checkpoints/pilot/dry_run.pt \
    --max_epochs 1 \
    --batch_size 32 \
    --learning_rate 2e-6 \
    --warmup_steps 1000 \
    --max_src_tokens 2500 \
    --eval_steps 1000 \
    --log_steps 100 \
    --mode SPEECH_TO_TEXT \
    --patience 10
    
    
    