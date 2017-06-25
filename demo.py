from utils import read_gray_img
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import predict_net
def demo(img_path):
	net = predict_net()
	net.load_state_dict(torch.load('checkpoint/crowd_net2.pth'))
	input_img = read_gray_img(img_path)
	input_img = torch.autograd.Variable(torch.Tensor(input_img/255.0))
	print(input_img.size())
	#input_image = input_image.view(1, 3, 255, 255)
	heat_map = net.forward(input_img)
	print heat_map.size()
	heat_map = torch.squeeze(heat_map)
	heat_map = heat_map.data.numpy()
	plt.imshow(heat_map, cmap = 'hot')
	plt.show()
if __name__ == '__main__':
	demo('demo/demo3.jpg')
