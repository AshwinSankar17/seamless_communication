# To start multi-node training, change nnodes to $NUM_NODES and rdzv-endpoint to $MASTER_ADDR:$MASTER_PORT in all nodes
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

## Get ip address of the machine using ifconfig. Grep for "inet " and get the first occurence. Then trim the string. Then split the string by space and get the second element.
ipaddr=$(ifconfig | grep "inet " | head -n 1 | xargs | cut -d ' ' -f 2)

num_nodes=$1
node_rank=$2

## if node rank is 0 then we write the master address and port to a file. If not we read it from the file
if [ $node_rank -eq 0 ]; then
    echo $ipaddr > master_addr
else
    ipaddr=$(cat master_addr)
fi

echo "Running on $num_nodes nodes with rank $node_rank and master addr $ipaddr"

OMP_NUM_THREADS=16 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nnodes=$num_nodes \
    --nproc_per_node=8 \
    --node_rank=$node_rank \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$ipaddr:8009 \
    --no-python \
    m4t_finetune \
    --train_dataset data/BhasaAnuvaad/tam/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/hindi/train_manifest.json \
    --eval_dataset data/BhasaAnuvaad/Mann-ki-Baat/indic2en/hindi/test_manifest.json \
    --model_name seamlessM4T_v2_large \
    --save_model_to models/checkpoints/pilot/dry_run.pt \
    --max_epochs 1 \
    --batch_size 32 \
    --learning_rate 2e-6 \
    --warmup_steps 1000 \
    --max_src_tokens 2500 \
    --eval_steps 10 \
    --log_steps 10 \
    --mode SPEECH_TO_TEXT \
    --patience 10
    
    
# OMP_NUM_THREADS=16 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nnodes=$num_nodes \
#     --nproc_per_node=8 \
#     --node_rank=$node_rank \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$ipaddr:8009 \
#     --no-python \
#     m4t_finetune \
#     --train_dataset data/BhasaAnuvaad/tam/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/hindi/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/tamil/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/odia/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/kannada/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/marathi/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/punjabi/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/urdu/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/gujarati/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/telugu/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/assamese/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/malayalam/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/bengali/train_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/en2indic/english/train_manifest.json data/BhasaAnuvaad/UGCE-Resources/en2indic/english/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/hindi/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/tamil/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/kannada/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/marathi/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/gujarati/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/telugu/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/assamese/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/malayalam/train_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/bengali/train_manifest.json data/BhasaAnuvaad/NPTEL/en2indic/english/train_manifest.json data/BhasaAnuvaad/guj/train_manifest.json data/BhasaAnuvaad/kan/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/hindi/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/tamil/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/odia/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/kannada/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/marathi/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/punjabi/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/nepali/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/gujarati/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/telugu/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/assamese/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/malayalam/train_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/bengali/train_manifest.json data/BhasaAnuvaad/snd/train_manifest.json data/BhasaAnuvaad/asm/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/hindi/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/tamil/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/odia/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/kannada/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/marathi/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/punjabi/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/urdu/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/gujarati/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/telugu/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/assamese/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/malayalam/train_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/bengali/train_manifest.json data/BhasaAnuvaad/Vanipedia/en2indic/english/train_manifest.json data/BhasaAnuvaad/tel/train_manifest.json data/BhasaAnuvaad/ory/train_manifest.json data/BhasaAnuvaad/eng/train_manifest.json data/BhasaAnuvaad/pan/train_manifest.json data/BhasaAnuvaad/mar/train_manifest.json data/BhasaAnuvaad/hin/train_manifest.json data/BhasaAnuvaad/mal/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/hindi/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/tamil/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/odia/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/kannada/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/marathi/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/punjabi/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/urdu/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/gujarati/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/telugu/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/malayalam/train_manifest.json data/BhasaAnuvaad/WordProject/indic2en/bengali/train_manifest.json data/BhasaAnuvaad/WordProject/en2indic/english/train_manifest.json data/BhasaAnuvaad/ben/train_manifest.json \
#     --eval_dataset data/BhasaAnuvaad/Mann-ki-Baat/indic2en/hindi/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/tamil/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/odia/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/kannada/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/marathi/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/punjabi/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/urdu/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/gujarati/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/telugu/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/assamese/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/malayalam/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/indic2en/bengali/test_manifest.json data/BhasaAnuvaad/Mann-ki-Baat/en2indic/english/test_manifest.json data/BhasaAnuvaad/UGCE-Resources/en2indic/english/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/hindi/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/tamil/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/kannada/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/marathi/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/gujarati/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/telugu/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/assamese/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/malayalam/test_manifest.json data/BhasaAnuvaad/NPTEL/indic2en/bengali/test_manifest.json data/BhasaAnuvaad/NPTEL/en2indic/english/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/hindi/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/tamil/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/odia/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/kannada/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/marathi/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/punjabi/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/nepali/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/gujarati/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/telugu/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/assamese/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/malayalam/test_manifest.json data/BhasaAnuvaad/Spoken-Tutorial/indic2en/bengali/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/hindi/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/tamil/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/odia/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/kannada/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/marathi/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/punjabi/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/urdu/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/gujarati/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/telugu/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/assamese/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/malayalam/test_manifest.json data/BhasaAnuvaad/IndicVoices-ST/indic2en/bengali/test_manifest.json data/BhasaAnuvaad/Vanipedia/en2indic/english/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/hindi/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/tamil/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/odia/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/kannada/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/marathi/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/punjabi/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/urdu/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/gujarati/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/telugu/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/malayalam/test_manifest.json data/BhasaAnuvaad/WordProject/indic2en/bengali/test_manifest.json data/BhasaAnuvaad/WordProject/en2indic/english/test_manifest.json \
#     --model_name seamlessM4T_v2_large \
#     --save_model_to models/checkpoints/pilot_full/dry_run_full.pt \
#     --max_epochs 2 \
#     --batch_size 32 \
#     --learning_rate 2e-6 \
#     --warmup_steps 1000 \
#     --max_src_tokens 2500 \
#     --eval_steps 10 \
#     --log_steps 10 \
#     --mode SPEECH_TO_TEXT \
#     --patience 10