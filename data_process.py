import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
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
from data_agumentation import *

def load_gt_from_json(gt_file, gt_shape):
        gt = np.zeros(gt_shape, dtype='uint8') 
        with open(gt_file, 'r') as jf:
            for j, dot in enumerate(json.load(jf)):
                try:
                    gt[int(math.floor(dot['y'])), int(math.floor(dot['x']))] = 1
                except IndexError:
                    print gt_file, dot['y'], dot['x'], sys.exc_info()
                    return gt

def load_images_and_gts(path):
    images = []
    gts = []
    densities = []
    for gt_file in glob.glob(os.path.join(path, '*.json')):
        print(gt_file)
        if os.path.isfile(gt_file.replace('.json', '.png')):
            img = cv2.imread(gt_file.replace('.json', '.png'))
        else:
            print(gt_file.replace('.json', '.jpg'))
            img = cv2.imread(gt_file.replace('.json', '.jpg'))
        images.append(img)
        gt = load_gt_from_json(gt_file, img.shape[:-1])

        gts.append(gt)

        density_file = gt_file.replace('.json', '.h5')
        if os.path.isfile(density_file):
            with h5py.File(density_file, 'r') as hf:
                density = np.array(hf.get('density'))
        else:
            density = gaussian_filter_density([gt])[0]
            with h5py.File(density_file, 'w') as hf:
                hf['density'] = density
        densities.append(density)
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
    return X_train, Y_train
def process_dump_tohdf5data(X,Y, path, phase):

    batch_size = 7000
    X_process = np.zeros((batch_size, 3, patch_h, patch_w), dtype = np.float32)
    Y_process = np.zeros((batch_size, net_density_h, net_density_w), dtype = np.float32)
    with open(os.path.join(path, phase+'.txt'), 'w') as f:
        i1 = 0
        while i1 < len(X):
            if i1+batch_size < len(X):
                i2 = i1 + batch_size
            else:
                i2 = len(X)
            file_name = os.path.join(path, phase+'_'+str(i1)+'.h5')
            with h5py.File(file_name, 'w') as hf:
                for j, img in enumerate(X[i1:i2]):
                    X_process[j] = img.copy().transpose(2,0,1).astype(np.float32)
                    Y_process[j] = density_resize(Y[i1+j], fx = float(net_density_w)/patch_w, fy = float(net_density_h) / patch_h)
                hf['data'] = X_process[:(i2-i1)]
                hf['label'] = Y_process[:(i2-i1)]
            f.write(file_name+'\n')
            i1 += batch_size
            


def test_density_gen():
    img_path = 'dataset/UCF_CC_50/1.jpg'
    load_one_image(img_path)
def main():
    dataset_path = ['dataset/UCF_CC_50/']
    X_train, Y_train = gen_train_data(dataset_path)
    save_path = 'dataset/processed_hdf5'
    process_dump_tohdf5data(X_train, Y_train, save_path, 'train')
if __name__ =='__main__':
    main()
