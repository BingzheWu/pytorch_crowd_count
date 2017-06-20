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
            img = cv2.imread(gt_file.replace('json', '.jpg'))
        images.append(img)

        gt = load_gt_from_json(gt_file, img.shape[:-1])

        gts.append(gt)

        density_file = gt_file.replace('.json', 'h5')
        if os.path.isfile(density_file):
            with h5py.File(density_file, 'r') as hf:
                density = np.array(hf.get('density'))
        else:
            density = gaussian_filter_density([gt])[0]
            with h5py.File(density_file, 'w') as hf:
                hf['density'] = density
            densities.append(density)
    return (images, gts, densitites)
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
def test_density_gen():
    img_path = 'dataset/UCF_CC_50/1.jpg'
    load_one_image(img_path)

if __name__ =='__main__':
    test_density_gen()
