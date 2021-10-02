#/bin/bash
# model_path: the pretrained model path
# test_choice: [HYBRID/RESIZE] the test strategy you want to use

dataset_choice='SAMPLES'
test_choice='HYBRID'
model_path='models/pretrained/aimnet_pretrained_matting.pth'

python core/test.py \
     --cuda \
     --dataset_choice=$dataset_choice \
     --model_path=$model_path\
     --test_choice=$test_choice \

