#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import math
import numpy as np
cfg = {
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}
def make_layers(cfg, batch_norm = False):
	layers = []
	in_channels = 3
	idx_M = 0 
	for v in cfg:
		if v == 'M':
			idx_M += 1
			if idx_M == 4:
				layers += [nn.MaxPool2d(kernel_size = 2, stride = 1)]
			else:
				layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
			else:
				layers += [conv2d, nn.ReLU(inplace = True)]
			in_channels = v
	return nn.Sequential(*layers)

class high_level_feature_net(nn.Module):

	def __init__(self):
		super(high_level_feature_net, self).__init__()
	def forward(self, x):
		feature = make_layers(cfg = cfg['D'], batch_norm = True)(x)
		return feature
class low_level_feature_net(nn.Module):
	def __init__(self):
		super(low_level_feature_net, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 24, kernel_size = 5, padding = 3, ),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 5, stride = 2),
			nn.Conv2d(24, 24, kernel_size = 5, padding = 3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 5, stride = 2),
			nn.Conv2d(24, 24, kernel_size = 5, padding = 3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 5, stride = 2))
	def forward(self, x):
		feature = self.features(x)
		return feature
class predict_net(nn.Module):
	def __init__(self):
		super(predict_net, self).__init__()
		self.conv2d = nn.Conv2d(536, 1, kernel_size = 1)
	def forward(self, x):
		low_level_feature = low_level_feature_net().forward(x)
		high_level_feature = high_level_feature_net().forward(x)
		print(high_level_feature.size())
		concat_feature = torch.cat((low_level_feature, high_level_feature), 1)
		heat_map = self.conv2d(concat_feature)
		return heat_map
if __name__ == '__main__':
	input_img = np.zeros((20, 3, 225, 225))
	input_img = torch.autograd.Variable(torch.Tensor(input_img))
	heat_map = predict_net().forward(input_img)
	print(heat_map.size())