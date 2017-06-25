import torch
import h5py
import glob
import os
import numpy as np
import torch.utils.data as data
import lmdb
class UCF_CC_50(data.Dataset):
	def __init__(self, lmdb_image_datapath, lmdb_label_datapath):
                super(UCF_CC_50, self).__init__()
		self.lmdb_image_datapath = lmdb_image_datapath
		self.lmdb_label_datapath = lmdb_label_datapath
		self.images = []
		self.gts = []
		self.total_patches = 0
		self.limits = []
		self.num_files = 0
		self.file_list = []
		self.env_image = lmdb.open(self.lmdb_image_datapath)
		self.env_label = lmdb.open(self.lmdb_label_datapath)
		self.txn_image = self.env_image.begin()
		self.txn_label = self.env_label.begin()
		self.cursor_image = iter(self.txn_image.cursor())
		self.cursor_label = self.txn_label.cursor()
	def __getitem__(self, index):
		key = '{:010}'.format(index)
		image = self.txn_image.get(key)
		image = np.fromstring(image, dtype = np.float32)
		image = image.reshape((3, 225, 225))
		label = self.txn_label.get(key)
		label = np.fromstring(label, dtype= np.float32).reshape((27, 27))
		return image/255.0, label 
	def __len__(self):
		return 32000
def dataloader(image_path, label_path, phase = 'train', batch_size = 20, num_workers = 1):
	dataset = UCF_CC_50(image_path, label_path)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers)
	return data_loader
if __name__ == '__main__':
	dataset = UCF_CC_50('dataset/processed_lmdb_image', 'dataset/processed_lmdb_label')
	print type(dataset.__getitem__(14000))
