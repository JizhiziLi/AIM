"""
Deep Automatic Natural Image Matting [IJCAI-21]
Base Configurations class.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/AIM
Paper link : https://www.ijcai.org/proceedings/2021/111

"""

##############################################################
# Some roots
REPOSITORY_ROOT_PATH = ''
DUTS_DATASET_ROOT_PATH = ''
DIM_DATASET_ROOT_PATH = ''
HATT_DATASET_ROOT_PATH = ''
AM2K_DATASET_ROOT_PATH = ''
BG20K_DATASET_ROOT_PATH = ''
AIM_DATASET_ROOT_PATH = ''
HATT_DIM_TYPE_JSON = REPOSITORY_ROOT_PATH+'dataset/dim_hatt_type.json'

TRAIN_LOGS_FOLDER = REPOSITORY_ROOT_PATH+'logs/train_logs/'
TEST_LOGS_FOLDER = REPOSITORY_ROOT_PATH+'logs/test_logs/'

######### Path of dataset related ######### 
DATASET_PATHS_DICT={
'SOD_DUTS':{
	'TRAIN':{
		'ROOT_PATH':DUTS_DATASET_ROOT_PATH+'DUTS-TR/',
		'ORIGINAL_PATH':DUTS_DATASET_ROOT_PATH+'DUTS-TR/DUTS-TR-Image/',
		'MASK_PATH':DUTS_DATASET_ROOT_PATH+'DUTS-TR/DUTS-TR-Mask/',
		'SAMPLE_NUMBER':10553,
		'SAMPLE_BAGS':1
			}
		},
'DIM':{
	'TRAIN':{
		'ROOT_PATH':DIM_DATASET_ROOT_PATH+'train/',
		'FG_PATH': DIM_DATASET_ROOT_PATH+'train/fg/',
		'MASK_PATH': DIM_DATASET_ROOT_PATH+'train/mask/',
		'USR_PATH': DIM_DATASET_ROOT_PATH+'train/usr/',
		'SAMPLE_NUMBER':431,
		'SAMPLE_BAGS':5
			}
		},
'HATT':{
	'TRAIN':{
		'ROOT_PATH':HATT_DATASET_ROOT_PATH+'train/',
		'FG_PATH': HATT_DATASET_ROOT_PATH+'train/fg/',
		'MASK_PATH': HATT_DATASET_ROOT_PATH+'train/mask/',
		'USR_PATH': HATT_DATASET_ROOT_PATH+'train/usr/',
		'SAMPLE_NUMBER':591,
		'SAMPLE_BAGS':5
			}
		},
'AM2K':{
	'TRAIN':{
		'ROOT_PATH':AM2K_DATASET_ROOT_PATH+'train/',
		'FG_PATH': AM2K_DATASET_ROOT_PATH+'train/fg/',
		'MASK_PATH': AM2K_DATASET_ROOT_PATH+'train/mask/', 
		'USR_PATH': AM2K_DATASET_ROOT_PATH+'train/usr/', 
		'SAMPLE_NUMBER':1800,
		'SAMPLE_BAGS':2
			}
		},
'AIM':{
	'TEST':{
		'ROOT_PATH':AIM_DATASET_ROOT_PATH,
		'ORIGINAL_PATH':AIM_DATASET_ROOT_PATH+'original/',
		'MASK_PATH':AIM_DATASET_ROOT_PATH+'mask/',
		'TRIMAP_PATH':AIM_DATASET_ROOT_PATH+'trimap/',
		'USR_PATH':AIM_DATASET_ROOT_PATH+'usr/',
		'SAMPLE_NUMBER':500,
		'SAMPLE_BAGS':1,
			}
		},
'BG20K':{
	'TRAIN':{
		'ROOT_PATH': BG20K_DATASET_ROOT_PATH,
		'ORIGINAL_PATH': BG20K_DATASET_ROOT_PATH+'train/',
			},
		},

}
JSON_AIM_CATEGORY_TYPE = AIM_DATASET_ROOT_PATH+'aim_category_type.json'
AIM_500_TYPE_LIST = ['SO','STM', 'NS']
AIM_500_CATEGORY_LIST = ['animal','fruit','furniture','plant','portrait','toy','transparent']
##############################################################
# Some parameter for data processing
MAX_SIZE_H = 1600
MAX_SIZE_W = 1600
SHORTER_PATH_LIMITATION=1080
CROP_SIZE = [640, 960, 1280]
RESIZE_SIZE = 320
##############################################################
# Path of samples images/results/transparent results
SAMPLES_ORIGINAL_PATH = REPOSITORY_ROOT_PATH+'samples/original/'
SAMPLES_RESULT_ALPHA_PATH = REPOSITORY_ROOT_PATH+'samples/result_alpha/'
SAMPLES_RESULT_COLOR_PATH = REPOSITORY_ROOT_PATH+'samples/result_color/'
# Path of the pretrained models
PRETRAINED_AIMNET_DUTS = REPOSITORY_ROOT_PATH+'models/pretrained/aimnet_pretrained_duts.pth'
PRETRAINED_AIMNET_MATTING = REPOSITORY_ROOT_PATH+'models/pretrained/aimnet_pretrained_matting.pth'
PRETRAINED_R34_MP = REPOSITORY_ROOT_PATH+'models/pretrained/r34mp_pretrained_imagenet.pth.tar'