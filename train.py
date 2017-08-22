import cv2
import os
import numpy as np 
import torch
import torch.nn.init as init
from models import predict_net
import torchvision.models as tmd
from loader import dataloader
import torch.nn as nn
def train():
	net = predict_net()
    	################################
    	vgg16 = tmd.vgg16(pretrained=True)
    	model_dict = net.state_dict()
    	pretrained_dict = vgg16.state_dict()
    	for name in model_dict:
        	if name == 'high_level_feature_net.features.0.weight':
            		model_dict[name] = pretrained_dict['features.0.weight']
		if name == 'high_level_feature_net.features.0.bias':
            		model_dict[name] = pretrained_dict['features.0.bias']
		if name == 'high_level_feature_net.features.3.weight':
            		model_dict[name] = pretrained_dict['features.2.weight']
		if name == 'high_level_feature_net.features.3.bias':
            		model_dict[name] = pretrained_dict['features.2.bias']
		if name == 'high_level_feature_net.features.7.weight':
            		model_dict[name] = pretrained_dict['features.5.weight']
		if name == 'high_level_feature_net.features.7.bias':
            		model_dict[name] = pretrained_dict['features.5.bias']
		if name == 'high_level_feature_net.features.10.weight':
            		model_dict[name] = pretrained_dict['features.7.weight']
		if name == 'high_level_feature_net.features.10.bias':
            		model_dict[name] = pretrained_dict['features.7.bias']
		if name == 'high_level_feature_net.features.14.weight':
            		model_dict[name] = pretrained_dict['features.10.weight']
		if name == 'high_level_feature_net.features.14.bias':
            		model_dict[name] = pretrained_dict['features.10.bias']
		if name == 'high_level_feature_net.features.17.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.20.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.24.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.27.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.30.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.34.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.36.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.39.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'high_level_feature_net.features.42.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		'''		
		if name == 'high_level_feature_net.features.17.weight':
            		model_dict[name] = pretrained_dict['features.12.weight']
		if name == 'high_level_feature_net.features.17.bias':
            		model_dict[name] = pretrained_dict['features.12.bias']
		if name == 'high_level_feature_net.features.20.weight':
            		model_dict[name] = pretrained_dict['features.14.weight']
		if name == 'high_level_feature_net.features.20.bias':
            		model_dict[name] = pretrained_dict['features.14.bias']
		if name == 'high_level_feature_net.features.24.weight':
            		model_dict[name] = pretrained_dict['features.17.weight']
		if name == 'high_level_feature_net.features.24.bias':
            		model_dict[name] = pretrained_dict['features.17.bias']
		if name == 'high_level_feature_net.features.27.weight':
            		model_dict[name] = pretrained_dict['features.19.weight']
		if name == 'high_level_feature_net.features.27.bias':
            		model_dict[name] = pretrained_dict['features.19.bias']
		'''
		if name == 'low_level_feature_net.features.0.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'low_level_feature_net.features.3.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
		if name == 'low_level_feature_net.features.6.weight':
			init.kaiming_uniform(model_dict[name], mode='fan_in')
    	net.load_state_dict(model_dict)
   	#################################
	data_loader = dataloader('dataset/ucf_lmdb_image', 'dataset/ucf_lmdb_label')
	criterion = nn.MSELoss().cuda()
	net = net.cuda()
	for module in net.modules():
		module.cuda()
		print(module)
	net.train()
	optimizer = torch.optim.SGD(net.parameters(), lr = 1e-5, momentum = 0.9)
	for epoch in range(10000):
		for i, data in enumerate(data_loader):
			if i>=201:
				break
			inputs, gts = data
			inputs, gts = torch.autograd.Variable(inputs), torch.autograd.Variable(gts)
			inputs = inputs.cuda()
			gts = gts.cuda()
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, gts)
			loss.backward()
			optimizer.step()
			running_loss  = loss.data[0]
			if i%50 == 0:
				print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
		torch.save(net.state_dict(), 'ucftrain1/crowd_net%d.pth'%(epoch))
            	#running_loss = 0.0
if __name__ == '__main__':
	train()
