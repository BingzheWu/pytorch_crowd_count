import cv2
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
def read_data(datapath, batch_size = 20, phase = 'train'):
	for h5file in os.listdir(datapath):
		if phase in h5file:
			print h5file
			h5file = os.path.join(datapath, h5file)
			print(h5file)
			with h5py.File(h5file, 'r') as hf:
				images = hf.get('data')
				gts = hf.get('label')
				print(type(images[:20]))
				tmp_data = np.array(images[:25])
				tmp_gt = np.array(gts[:25])
				print(tmp_gt.shape)
				#vis_square(tmp_gt, mode = 'heat')
				#vis_square(list_to_np_array(tmp_data.transpose((0,2,3,1))))
		break
def list_to_np_array(in_list):
    max_h = 0
    max_w = 0
    for i, item in enumerate(in_list):
        if item.shape[0] > max_h:
            max_h = item.shape[0]
        if item.shape[1] > max_w:
            max_w = item.shape[1]
    out_arr = np.zeros((len(in_list), max_h, max_w, 3))
    for i, item in enumerate(in_list):
        pad_h = max_h - item.shape[0]
        pad_w = max_w - item.shape[1]
        out_arr[i] = np.pad(item, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0.)
    return out_arr

def vis_square(data, mode = None):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.figure(figsize=(30,30))
    plt.imshow(data)
    if mode == 'heat':
    	plt.imshow(data, cmap = 'hot')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
	datapath = 'dataset/processed_hdf5/'
	read_data(datapath)