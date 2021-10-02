##############################################################
# Set up some configurations here
import sys
import torch
##############################################################
# Some roots
AIM_DATASET_ROOT_PATH = ''
REPOSITORY_ROOT_PATH = ''
######### Path of dataset related ######### 
DATASET_PATHS_DICT={
'AIM_TEST':{
	'ROOT_PATH':AIM_DATASET_ROOT_PATH+'validation/',
	'ORIGINAL_PATH':AIM_DATASET_ROOT_PATH+'validation/original/',
	'MASK_PATH':AIM_DATASET_ROOT_PATH+'validation/mask/',
	'TRIMAP_PATH':AIM_DATASET_ROOT_PATH+'validation/trimap/',
	'SAMPLE_NUMBER':500,
},}
JSON_AIM_CATEGORY_TYPE = AIM_DATASET_ROOT_PATH+'aim_category_type.json'
AIM_500_TYPE_LIST = ['SO','STM', 'NS']
AIM_500_CATEGORY_LIST = ['animal','fruit','furniture','plant','portrait','toy','transparent']
##############################################################
# Path for logging in training/testing
TRAIN_LOGS_FOLDER = REPOSITORY_ROOT_PATH+'logs/train_logs/'
TEST_LOGS_FOLDER = REPOSITORY_ROOT_PATH+'logs/test_logs/'
##############################################################
# Some parameter for data processing
MAX_SIZE_H = 1600
MAX_SIZE_W = 1600
SHORTER_PATH_LIMITATION=1080
AIM_SIZE_H = 320
AIM_SIZE_W = 320
AIM_CROP_H = [640, 960, 1280]
AIM_CROP_W = [640, 960, 1280]
AIM_BS = 16
E2E_SOD_H = 320
E2E_SOD_W = 320
##############################################################
# Path of samples images/results/transparent results
SAMPLES_ORIGINAL_PATH = REPOSITORY_ROOT_PATH+'samples/original/'
SAMPLES_RESULT_ALPHA_PATH = REPOSITORY_ROOT_PATH+'samples/result_alpha/'
SAMPLES_RESULT_COLOR_PATH = REPOSITORY_ROOT_PATH+'samples/result_color/'
# Path of the pretrained models
PRETRAINED_AIMNET_DUTS = REPOSITORY_ROOT_PATH+'models/pretrained/aimnet_pretrained_duts.pth'
PRETRAINED_AIMNET_MATTING = REPOSITORY_ROOT_PATH+'models/pretrained/aimnet_pretrained_matting.pth'
PRETRAINED_R34_MP = REPOSITORY_ROOT_PATH+'models/pretrained/r34mp_pretrained_imagenet.pth.tar'