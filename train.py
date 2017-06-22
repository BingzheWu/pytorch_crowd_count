import os
import numpy as np 
import torch
from models import predict_net
from loader import dataloader
import torch.nn as nn
def train():
	net = predict_net()
	data_loader = dataloader('dataset/processed_hdf5')
	criterion = nn.MSELoss().cuda()
	net = net.cuda()
	for module in net.modules():
		module.cuda()
		print(module)
	net.train()
	optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
	for epoch in range(10):
		for i, data in enumerate(data_loader, 0):
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
			if i%100 == 0:
				print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
		torch.save(model.state_dict(), 'checkpoint/crowd_net%d.pth'%(epoch))
            	#running_loss = 0.0
if __name__ == '__main__':
	train()