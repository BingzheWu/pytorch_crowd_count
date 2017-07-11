import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as scio
import scipy
from skimage.transform import downscale_local_mean
import os
import sys
import json
import math
import time
import random
from random import shuffle
import pickle
import h5py
import glob
import lmdb
from data_agumentation import *
'''
# this one for ShanghaiTech
def load_gt_from_mat(gt_file, gt_shape):
    gt = np.zeros(gt_shape, dtype='uint8') 
    gt_dict = scio.loadmat(gt_file)
    dict_key = 'image_info'
    struct = gt_dict[dict_key]
    gt_mat = struct[0][0][0][0][0]
    for pix in gt_mat:
                try:
                    gt[int(math.floor(pix[1])), int(math.floor(pix[0]))] = 1
                except IndexError:
                    print gt_file, pix[1], pix[0], sys.exc_info()
    return gt
'''

# this one for ucf_cc_50
def load_gt_from_mat(gt_file, gt_shape):
	gt = np.zeros(gt_shape, dtype='uint8')
	gt_dict = scio.loadmat(gt_file)
	dict_key = 'annPoints'
	gt_mat = gt_dict[dict_key]
	for pix in gt_mat:
		try:
			gt[int(math.floor(pix[1])), int(math.floor(pix[0]))] = 1
		except IndexError:
			print gt_file, pix[1], pix[0], sys.exc_info()
	return gt


def load_images_and_gts(path):
    images = []
    gts = []
    densities = []
    cnt = 1
    for gt_file in glob.glob(os.path.join(path, '*.mat')):
        print(gt_file)
        if os.path.isfile(gt_file.replace('.mat', '.png')):
            img = cv2.imread(gt_file.replace('.mat', '.png'))
        else:
            print(gt_file.replace('.mat', '.jpg'))
            img = cv2.imread(gt_file.replace('.mat', '.jpg'))
        images.append(img)
        gt = load_gt_from_mat(gt_file, img.shape[:-1])

        gts.append(gt)
        density_file = gt_file.replace('.mat', '.h5')
        if os.path.isfile(density_file):
	    print(density_file)
            with h5py.File(density_file, 'r') as hf:
                density = np.array(hf.get('density'))
        else:
            density = gaussian_filter_density([gt])[0]
            with h5py.File(density_file, 'w') as hf:
                hf['density'] = density
        densities.append(density)
	print("################################################")
	print(cnt)
	print("################################################")
	cnt += 1
    return (images, gts, densities)
def load_one_image(img_path):
    gt_file = img_path.replace('.jpg', '.json')
    img = cv2.imread(img_path)
    gt = load_gt_from_json(gt_file, img.shape[:-1])
    density = gaussian_filter_density([gt])[0]
    import matplotlib.pyplot as plt
    plt.imshow(density, cmap = 'hot')
    plt.show()
    print(density.shape)

def gaussian_filter_density(gts):
    densities = []
    for gt in gts:
        print(gt.shape)
        density = np.zeros(gt.shape, dtype = np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density
        pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
        leafsize = 2048

        tree = scipy.spatial.KDTree(pts.copy(), leafsize = leafsize)
        distances, locations = tree.query(pts, k = 2, eps = 10.)

        for i, pt in enumerate(pts):
	    print(i)
            pt2d = np.zeros(gt.shape, dtype = np.float32)
            pt2d[pt[1], pt[0]] = 1
            if gt_count >1:
                sigma = distances[i][1]
            else:
                sigma = np.average(np.array(gt.shape))/2./2.
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode = 'constant')
    densities.append(density)
    return densities
## train batch gen
def gen_train_data(dataset_paths):
    X_fs = []
    Y_fs = []

    for path in dataset_paths:
        images, gts, densities = load_images_and_gts(path)
        X_fs += images
        Y_fs += densities
    from sklearn.model_selection import train_test_split
    X_fs_train, X_fs_test, Y_fs_train, Y_fs_test = train_test_split(X_fs, Y_fs, test_size = 0.2)
    X_train, Y_train = X_fs_train, Y_fs_train
    X_test, Y_test = X_fs_test, Y_fs_test
    print(len(X_train))
    X_train, Y_train = multiscale_pyramidal(X_train, Y_train)
    #X_train, Y_train = adapt_images_and_densities(X_train, Y_train, slice_w, slice_h)
    print(len(X_train))
    X_train, Y_train = generate_slices(X_train, Y_train, slice_w = patch_w, slice_h = patch_h, offset = 8)
    print(len(X_train))
    #X_train, Y_train = crop_slices(X_train, Y_train)
    X_train, Y_train = flip_slices(X_train, Y_train)
    print(len(X_train))
    X_train, Y_train = samples_distribution(X_train,Y_train)
    print(len(X_train))
    X_train,Y_train = shuffle_slices(X_train, Y_train)
    print('gen is ok')
    x = X_train[156]
    y = Y_train[156]
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    return X_train, Y_train


def read_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path)
    with env.begin() as txn:
        cursor = txn.cursor()
        for (idx, (key, value)) in enumerate(cursor):
            image = np.fromstring(value, dtype = np.float32)
            #image = np.reshape(image, (3,225,225))/255.0
            image = np.reshape(image, (27, 27))
            #image = image.transpose((1,2,0))
            print(image)
            plt.imshow(image, cmap = 'hot')
            plt.show()
            break
    #image = txn.get('0')
    #image = np.fromstring(image)[0]
    #print image.shape

def lmdb_create(image_save_path, label_save_path, images, gts):
    map_size = 1e12
    env_image = lmdb.open(image_save_path, map_size = map_size)
    env_label = lmdb.open(label_save_path, map_size = map_size)
    idx = 0
    txn_image = env_image.begin(write = True, buffers = True)
    txn_label = env_label.begin(write = True, buffers = True)
    images = np.array(images)
    gts = np.array(gts)
    for i in range(images.shape[0]):
        image = images[i]
        gt = gts[i]
        image = image.copy().transpose(2,0,1).astype(np.float32)
        gt = density_resize(gt, fx = float(net_density_w)/patch_w, fy = float(net_density_h) / patch_h)                
        image = image.tostring()
        label = gt.tostring()
        str_id = '{:010}'.format(idx)
        #print(data.shape)                    
        txn_image.put(str_id.encode('ascii'), image)
        txn_label.put(str_id.encode('ascii'), label)
        if idx % 100 == 0:
            txn_image.commit()
            txn_label.commit()
            txn_label = env_label.begin(write = True, buffers = True)
            txn_image = env_image.begin(write = True, buffers = True)
        idx += 1
    txn_image.commit()
    txn_label.commit()






def test_density_gen():
    img_path = 'dataset/UCF_CC_50/1.jpg'
    load_one_image(img_path)
def main():
	dataset_path = ['dataset/UCF_CC_50']
	X_train, Y_train = gen_train_data(dataset_path)
	image_save_path = 'dataset/ucf_lmdb_image'
	label_save_path = 'dataset/ucf_lmdb_label'
	lmdb_create(image_save_path, label_save_path, X_train, Y_train)
	


if __name__ =='__main__':
    main()
