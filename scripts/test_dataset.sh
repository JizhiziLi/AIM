#/bin/bash
# nickname: your test logging file along with evaluation results 
# will be in the file `logs/test_logs/nickname.log`
# test_choice: [HYBRID/RESIZE] the test strategy you want to use
# model_path: the pretrained model path
# test_result_dir: the path to save the predict results


dataset_choice='AIM_500'
test_choice='HYBRID'
model_path='models/pretrained/aimnet_pretrained_matting.pth'
nickname=DEBUG

python core/test.py \
     --cuda \
     --dataset_choice=$dataset_choice \
     --model_path=$model_path\
     --test_choice=$test_choice \
     --test_result_dir=result/$nickname/ \
     --logname=$nickname \
     --deploy \
