#!/bin/bash

SOURCE=/home/asr/speech-datasets/indicvoices/train-data
DEST=/home/asr/speech-datasets/indicvoices/artifacts/manifests/train/
SCRIPT_PATH=/home/asr/speech-datasets/indicvoices/scripts

rm -r $DEST

mkdir -p ${DEST}/normalized
for l in $(ls ${SOURCE} --ignore 'manifests')
do
    sleep 2
    touch ${DEST}/normalized/train_${l,,}_indicvoices.json
    find ${SOURCE}/${l} -type f -wholename "*/train/*.json" -exec cat {} >> ${DEST}/normalized/train_${l,,}_indicvoices.json \; &

    sleep 2
    touch ${DEST}/normalized/valid_${l,,}_indicvoices.json
    find ${SOURCE}/${l} -type f -wholename "*/valid/*.json" -exec cat {} >> ${DEST}/normalized/valid_${l,,}_indicvoices.json \; &
    
done