#/bin/bash

batchsizePerGPU=16
GPUNum=1
batchsize=`expr $batchsizePerGPU \* $GPUNum`
threads=8
nEpochs=50
lr=0.000001
train_data='matting'
nickname=aim_matting

python core/train.py \
    --logname=$nickname \
    --batchSize=$batchsize \
    --threads=$threads \
    --nEpochs=$nEpochs \
    --lr=$lr \
    --train_data=$train_data \
    --model_save_dir=models/trained/$nickname/ \
    --pretrain=models/pretrained/aimnet_pretrained_duts.pth \