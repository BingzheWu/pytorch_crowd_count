import cv2
import numpy as np
import random
slice_w = 256
slice_h = 256
patch_w = 225
patch_h = 225
net_density_h = 28
net_density_w = 28
def density_resize(density, fx, fy):
    return cv2.resize(density, None, fx = fx, fy = fy, interpolation = cv2.INTER_CUBIC)/(fx*fy)
def multiscale_pyramidal(images, gts, start = 0.5, end = 1.3, step = 0.1):
    frange = np.arange(start, end, step)
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        for f in frange:
            out_images.append(cv2.resize(img, None, fx = f, fy = f, interpolation = cv2.INTER_CUBIC))
            out_gts.append(density_resize(gts[i], fx = f, fy = f))
    return (out_images, out_gts)
def adapt_images_and_densities(images, gts, slice_w = slice_w, slice_h = slice_h):
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        img_h, img_w, _ = img.shape
        n_slices_h = int(round(img_h/slice_h))
        n_slices_w = int(round(img_w/slice_w))
        new_img_h = float(n_slices_h *slice_h)
        new_img_w = float(n_slices_w*slice_w)
        fx = new_img_w / img_w
        fy = new_img_h/img_h
        out_images.append(cv2.resize(img, None, fx = fx, fy = fy, interpolation = cv2.INTER_CUBIC))
        assert out_images[-1].shape[0]%slice_h == 0 and out_images[-1].shape[1]%slice_w == 0
        if gts is not None:
            out_gts.append(density_resize(gts[i], fx, fy))
    return (out_images, out_gts)

def generate_slices(images, gts, slice_w = slice_w, slice_h = slice_h, offset = None):
    if offset == None:
        offset = slice_w
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        img_h, img_w,_ = img.shape
        p_y_id = 0
        p_y1 = 0
        p_y2 = p_y1 +slice_h
        while p_y2 <= img_h:
            p_x_id = 0
            p_x1 = 0
            p_x2 = p_x1+slice_w
            while p_x2 <= img_w:
                out_images.append(img[p_y1:p_y2, p_x1:p_x2])
                assert out_images[-1].shape[:-1] == (slice_h, slice_w)
                if gts is not None:
                    out_gts.append(gts[i][p_y1:p_y2, p_x1:p_x2])
                    assert out_gts[-1].shape == (slice_h, slice_w)
                p_x_id += 1
                p_x1 += offset
                p_x2 += offset
            p_y_id += 1
            p_y1 += offset
            p_y2 += offset
    return (out_images, out_gts)
def crop_slices(images, gts):
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        img_h, img_w, _ = img.shape
        gt = gts[i]
        p_y1, p_y2 = 0, patch_h
        p_x1, p_x2 = 0, patch_w
        out_images.append(img[p_y1:p_y2,p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
        ## top right
        p_y1,p_y2 = 0, patch_h
        p_x1, p_x2 = img_w - patch_w , img_w
        out_images.append(img[p_y1:p_y2,p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])

        ## bottom left
        p_y1, p_y2 = img_h - patch_h, img_h
        p_x1, p_x2 = 0, patch_w
        out_images.append(img[p_y1:p_y2,p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])

        ## bottom right

        p_y1, p_y2 = img_h - patch_h, img_h
        p_x1, p_x2 = img_w - patch_w, img_w
        out_images.append(img[p_y1:p_y2,p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
        ## center
        p_y1, p_y2 = int((img_h-patch_h)/2), int((img_h-patch_h)/2)+patch_h
        p_x1, p_x2 = int((img_w-patch_w)/2), int((img_w-patch_w)/2)+patch_w
        out_images.append(img[p_y1:p_y2,p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
    return (out_images, out_gts)

def flip_slices(images, gts):
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        img_h, img_w, _ = img.shape
        gt = gts[i]
        out_images.append(img)
        out_gts.append(gt)
        out_images.append(np.fliplr(img))
        out_gts.append(np.fliplr(gt))
    return (out_images, out_gts)
def shuffle_slices(images, gts):
    out_images = []
    out_gts = []
    index_shuf = range(len(images))
    random.shuffle(index_shuf)
    for i in index_shuf:
        out_images.append(images[i])
        out_gts.append(gts[i])
    return (out_images, out_gts)
def samples_distribution(images, gts):
    out_images = []
    out_gts = []
    gts_count = map(np.sum, gts)
    max_count = max(gts_count)
    for i, img in enumerate(images):
        if gts_count[i] >= 1. and random.random() < gts_count[i]**2/max_count**2:
            out_images.append(img)
            out_gts.append(gts[i])
    neg_count = sum(gt_count < 1. for gt_count in gts_count)
    obj_neg_count = len(out_gts) / 6 # ~= 15-16%
    neg_keep_prob = min(1., float(obj_neg_count) / float(neg_count))
    for i, img in enumerate(images):
        if gts_count[i] < 1. and random.random() < neg_keep_prob:
            out_images.append(img)
            out_gts.append(gts[i])
        
    return (out_images, out_gts)
