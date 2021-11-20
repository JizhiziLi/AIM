"""
Deep Automatic Natural Image Matting [IJCAI-21]
Dataset processing.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/AIM
Paper link : https://www.ijcai.org/proceedings/2021/111

"""
from config import *
from util import *
import torch
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import logging
from torchvision import transforms
from torch.autograd import Variable
from skimage.transform import resize


#########################
## Data transformer
#########################
class DatasetTransform(object):
	def __init__(self):
		super(DatasetTransform, self).__init__()

	def __call__(self, SOD_DATA, *argv):
		ori = argv[0]
		h, w, c = ori.shape
		rand_ind = random.randint(0, len(CROP_SIZE) - 1)
		crop_size = CROP_SIZE[rand_ind] if CROP_SIZE[rand_ind]<min(h, w) else 320
		resize_size = RESIZE_SIZE
		flip_flag=True if random.random()<0.5 else False

		if SOD_DATA:
			argv_transform = []
			for item in argv:
				if flip_flag:
					item = cv2.flip(item, 1)
				item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
				argv_transform.append(item)
		else:
		### generate crop centered in transition area randomly
			trimap = argv[2]
			trimap_crop = trimap[:h-crop_size, :w-crop_size]
			target = np.where(trimap_crop == 128) if random.random() < 0.5 else np.where(trimap_crop > -100)
			if len(target[0])==0:
				target = np.where(trimap_crop > -100)
			rand_ind = np.random.randint(len(target[0]), size = 1)[0]
			cropx, cropy = target[1][rand_ind], target[0][rand_ind]
			# generate samples (crop, flip, resize)
			argv_transform = []
			for item in argv:
				item = item[cropy:cropy+crop_size, cropx:cropx+crop_size]
				if flip_flag:
					item = cv2.flip(item, 1)
				item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
				argv_transform.append(item)
		return argv_transform

#########################
## Data Loader
#########################
class LoadDataset(torch.utils.data.Dataset):
	def __init__(self, args, transform):

		self.samples=[]
		self.logging = args.logging
		self.transform = transform
		self.SOD_DATA = True if args.train_data=='sod' else False
		
		self.logging.info('===> Loading training set')
		if self.SOD_DATA:
			self.samples += generate_paths_for_dataset(args,'SOD_DUTS')
		else:
			self.samples += generate_paths_for_dataset(args,'DIM')
			self.samples += generate_paths_for_dataset(args,'HATT')
			self.samples += generate_paths_for_dataset(args,'AM2K')

		self.logging.info(f"\t--crop_size: {CROP_SIZE} | resize: {RESIZE_SIZE}")
		self.logging.info("\t--Valid Samples: {}".format(len(self.samples)))

	def __getitem__(self,index):

		if self.SOD_DATA:
			ori_path, mask_path = self.samples[index]
			ori = np.array(Image.open(ori_path))
			mask = trim_img(np.array(Image.open(mask_path)))
			trimap = gen_trimap_with_dilate(mask, 10)
			argv = self.transform(True, ori, mask, trimap)
		else:
			fg_path, bg_path, mask_path, usr_path = self.samples[index]
			fg = np.array(Image.open(fg_path))
			bg = np.array(Image.open(bg_path))
			mask = trim_img(np.array(Image.open(mask_path)))
			ori, fg, bg = generate_composite(fg, bg, mask)
			usr = trim_img(np.array(Image.open(usr_path)))
			argv = self.transform(False, ori, mask, usr, fg, bg)

		argv_transform = []
		for item in argv:
			if item.ndim<3:
				item = Variable(torch.from_numpy(item.astype(np.float32)[np.newaxis, :, :]))
			else:
				item = Variable(torch.from_numpy(item.astype(np.float32)).permute(2, 0, 1))
			argv_transform.append(item)

		if self.SOD_DATA:
			[ori, mask, trimap] = argv_transform
			return ori, mask, trimap
		else:
			[ori, mask, usr, fg, bg] = argv_transform

			return ori, mask, usr, fg, bg

	def __len__(self):
		return len(self.samples)
