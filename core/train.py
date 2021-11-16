"""
Deep Automatic Natural Image Matting [IJCAI-21]
Main train file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/AIM
Paper link : https://www.ijcai.org/proceedings/2021/111

"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import logging
import numpy as np
import datetime
import time
from config import *
from util import *
from evaluate import *
from network.AimNet import AimNet
from data import LoadDataset, DatasetTransform

######### Parsing arguments ######### 
def get_args():
	parser = argparse.ArgumentParser(description='Arguments for the training purpose.')
	# train_data: choose either sod or matting as the training data.
	# model_save_dir: path to save the last checkpoint
	# logname: name of the logging files
	parser.add_argument('--logname', type=str, default='train_log', help="name of the logging file")
	parser.add_argument('--gpuNums', type=int, default=1, help='number of gpus')
	parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
	parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
	parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for, 500 for ORI-Track and 100 for COMP-Track')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.00001')	
	parser.add_argument('--train_data', type=str, required=True, default='matting', choices=["sod","matting"], help="options to generate dataset in training")
	parser.add_argument('--model_save_dir', type=str, help="where to save the final model")
	parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")

	args = parser.parse_args()
	print(args)
	return args

def load_dataset(args):
	train_transform = DatasetTransform()
	train_set = LoadDataset(args, train_transform)
	train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
	return train_loader

def load_model(args):
	model = AimNet()      
	start_epoch = 1
	if args.pretrain and os.path.isfile(args.pretrain):
		args.logging.info("===> loading pretrain '{}'".format(args.pretrain))
		ckpt = torch.load(args.pretrain)
		model.load_state_dict(ckpt['state_dict'],strict=True)

	return model, start_epoch

def format_second(secs):
	h = int(secs / 3600)
	m = int((secs % 3600) / 60)
	s = int(secs % 60)
	ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h,m,s)
	return ss    

def train(args, model, optimizer, train_loader, epoch):
	model = torch.nn.DataParallel(model).cuda()
	model.train()
	t0 = time.time()

	loss_each_epoch=[]
	args.logging.info("===============================")
	for iteration, batch in enumerate(train_loader, 1):
		torch.cuda.empty_cache()

		if iteration>4:
			return 0

		batch_new = []
		for item in batch:
			item = item.cuda()
			batch_new.append(item)
		if args.train_data == 'sod':
			[ori, mask, trimap] = batch_new
		else:
			[ori, mask, usr, fg, bg] = batch_new

		optimizer.zero_grad()
		predict_global, predict_local, predict_fusion = model(ori)
		predict_fusion = predict_fusion.cuda()
		
		if args.train_data=='sod':
			loss_global =get_crossentropy_loss(trimap, predict_global)
			loss_local = get_alpha_loss(predict_local, mask, trimap)
			loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask)
			loss = loss_global+loss_local+loss_fusion_alpha
		else:
			loss_global =get_crossentropy_loss(usr, predict_global)
			loss_local = get_alpha_loss(predict_local, mask, usr) + get_laplacian_loss(predict_local, mask, usr)
			loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask) + get_laplacian_loss_whole_img(predict_fusion, mask)
			loss_fusion_comp = get_composition_loss_whole_img(ori, mask, fg, bg, predict_fusion)
			loss = loss_global+loss_local+loss_fusion_alpha+loss_fusion_comp

		loss.backward()
		optimizer.step()

		if iteration !=  0:
			t1 = time.time()
			num_iter = len(train_loader)
			speed = (t1 - t0) / iteration
			exp_time = format_second(speed * (num_iter * (args.nEpochs - epoch + 1) - iteration))          
			loss_each_epoch.append(loss.item())
			if args.train_data=='sod':
				args.logging.info("AIM-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Global:{:.5f} Local:{:.5f} Fusion-alpha:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.item(), loss_global.item(), loss_local.item(), loss_fusion_alpha.item(), speed, exp_time))
			else:
				args.logging.info("AIM-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Global:{:.5f} Local:{:.5f} Fusion-alpha:{:.5f} Fusion-comp:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.item(), loss_global.item(), loss_local.item(), loss_fusion_alpha.item(), loss_fusion_comp.item(),speed, exp_time))
			
def save_last_checkpoint(args, model):
	args.logging.info('=====> Saving best model',str(args.epoch))
	create_folder_if_not_exists(args.model_save_dir)
	model_out_path = "{}ckpt_epoch{}.pth".format(args.model_save_dir, args.epoch)
	torch.save({'state_dict':model.state_dict()}, model_out_path)
	args.logging.info("Checkpoint saved to {}".format(model_out_path))

def main():
	args = get_args()
	now = datetime.datetime.now()
	# logging_filename = 'logs/train_logs/'+args.logname+'_'+now.strftime("%Y-%m-%d-%H:%M")+'.log'
	logging_filename = 'logs/train_logs/debug.log'
	print(f'===> Logging to {logging_filename}') 
	logging.basicConfig(filename=logging_filename, level=logging.INFO)
	args.logging = logging
	logging.info("===============================")
	logging.info(f"===> Loading args\n{args}")
	logging.info("===> Environment init")
	if not torch.cuda.is_available():
		raise Exception("No GPU and cuda available, please try again")
	
	args.gpuNums = torch.cuda.device_count()
	logging.info(f'Running with GPUs and the number of GPUs: {args.gpuNums}')
	train_loader = load_dataset(args)
	logging.info('===> Building the model')
	model, start_epoch = load_model(args)
	logging.info('===> Initialize optimizer')
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
	now = datetime.datetime.now()
	# training
	for epoch in range(start_epoch, args.nEpochs + 1):
		print(f'Train on Epoch: {epoch}')
		train(args, model, optimizer, train_loader, epoch)
		args.epoch = epoch

	save_last_checkpoint(args, model)

if __name__ == "__main__":
	main()