#/bin/bash

batchsizePerGPU=16
GPUNum=1
batchsize=`expr $batchsizePerGPU \* $GPUNum`
threads=8
nEpochs=100
lr=0.0001
train_data='sod'
nickname=aim_transfer_duts

python core/train.py \
    --logname=$nickname \
    --batchSize=$batchsize \
    --threads=$threads \
    --nEpochs=$nEpochs \
    --lr=$lr \
    --train_data=$train_data \
    --model_save_dir=models/trained/$nickname/ \