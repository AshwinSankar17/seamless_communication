#!/bin/bash

#SBATCH --job-name=m4t_finetune       # Job name
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=16            # CPUs per task
#SBATCH --gres=gpu:4                  # GPUs per node
#SBATCH --time=24:00:00               # Max runtime
#SBATCH --partition=your_partition    # Partition name

# Set variables
NUM_NODES=${SLURM_NNODES}
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
MASTER_PORT=8009  # Ensure this port is free across nodes
NPROC_PER_NODE=4  # GPUs per node

# Load modules or environment (if required)
module load pytorch

# Run multi-node training
srun torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    m4t_finetune \
    --train_dataset /data/BhasaAnuvaad/NPTEL/indic2en/assamese/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/en2indic/english/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/bengali/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/gujarati/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/hindi/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/kannada/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/malayalam/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/marathi/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/tamil/train_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/telugu/train_manifest.json \
    --eval_dataset /data/BhasaAnuvaad/NPTEL/indic2en/assamese/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/en2indic/english/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/bengali/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/gujarati/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/hindi/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/kannada/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/malayalam/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/marathi/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/tamil/val_manifest.json \
        /data/BhasaAnuvaad/NPTEL/indic2en/telugu/val_manifest.json \
    --model_name seamlessM4T_v2_large \
    --save_model_to /root/repos/seamless_communication/checkpoints/pilot/dry_run.pt \
    --max_epochs 100 \
    --batch_size 32 \
    --learning_rate 2e-6 \
    --warmup_steps 1000 \
    --max_src_tokens 2500 \
    --eval_steps 1000 \
    --log_steps 100 \
    --mode SPEECH_TO_TEXT \
    --patience 10
