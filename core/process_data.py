from util import *
from config import *
from tqdm import tqdm
import cv2
import numpy as np
import json
from PIL import Image


##############################
## Function gen_usr_from_mask 
## is aimed to generate USR(Unified Semantic Representataion) 
## from the ground truth label of datasets HATT, AIM, and AM2K.
##############################
def gen_usr_from_mask(mask_folder, usr_folder, kernel_size, dataset_choice):

    print(f'========>\n Generate USR(Unified Semantic Representation) for dataset: {dataset_choice}')
    print(f'========>\n Save results to the path: {usr_folder}')
    with open(HATT_DIM_TYPE_JSON) as f:
        hatt_dim_type = json.load(f)

    refresh_folder(usr_folder)
    mask_list = listdir_nohidden(mask_folder)

    for name in tqdm(mask_list):
        img_type = hatt_dim_type[dataset_choice][extract_pure_name(name)]
        mask = trim_img(np.array(Image.open(mask_folder+name)))
        usr = gen_trimap_with_dilate(mask, kernel_size)
        if img_type == 'STM':
            usr[usr>128]=128
        elif img_type == 'NS':
            usr[usr>-1]=128
        cv2.imwrite(usr_folder+name, usr)


def process_data():
    dataset_choice = 'DIM'
    kernel_size = 30
    mask_folder = DATASET_PATHS_DICT[dataset_choice]['TRAIN']['MASK_PATH']
    usr_folder = DATASET_PATHS_DICT[dataset_choice]['TRAIN']['USR_PATH']
    gen_usr_from_mask(mask_folder, usr_folder, kernel_size, dataset_choice)

    dataset_choice = 'HATT'
    kernel_size = 30
    mask_folder = DATASET_PATHS_DICT[dataset_choice]['TRAIN']['MASK_PATH']
    usr_folder = DATASET_PATHS_DICT[dataset_choice]['TRAIN']['USR_PATH']
    gen_usr_from_mask(mask_folder, usr_folder, kernel_size, dataset_choice)

    dataset_choice = 'AM2K'
    kernel_size = 30
    mask_folder = DATASET_PATHS_DICT[dataset_choice]['TRAIN']['MASK_PATH']
    usr_folder = DATASET_PATHS_DICT[dataset_choice]['TRAIN']['USR_PATH']
    gen_usr_from_mask(mask_folder, usr_folder, kernel_size, dataset_choice)

if __name__ == '__main__':
    process_data()