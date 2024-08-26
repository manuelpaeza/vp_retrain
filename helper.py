#!/usr/bin/env python

import colorsys
import numpy as np 
import random
from skimage.color import rgb2gray, gray2rgb
from skimage.draw import polygon2mask
import torch
import torchvision
from torchvision.tv_tensors import Mask, BoundingBoxes
from torchvision.transforms.v2 import functional as F

class ScaleImage:
    """
    Scale image so it is between 0-1: works on floats only

    Note normalize is (x-mean)/std
    You can use this to get data in [0,1] if you define
    mean as x.min(), and std as x.max()-x.min()  

    This was suggested by Nicolas Hug, see: https://github.com/pytorch/vision/issues/6753#issuecomment-1884978269
    """
    def __call__(self, img):
        min_val = img.min()
        max_val = img.max()
        range = max_val - min_val
        return F.normalize_image(img, mean=[min_val, min_val, min_val], std=[range, range, range])
    
def vp_load_image(dir, fnames, ind):
    """
    np.load image from directory, given list of fnames, and index   

    Input:
    dir: directory containing images
    fnames: list of filenames, sorted
    ind: index of desired image

    Returns:
    image
    full path to file
    """
    fname = fnames[ind]
    return np.load(dir + fname)['img'], dir + fname

def bounding_box(mask_coords):
    """
    note coords are in y/x!
    """
    x_vals = mask_coords[:,1]
    y_vals = mask_coords[:,0]
    return [(min(x_vals), min(y_vals)), (max(x_vals), max(y_vals))]

def create_mask(shape, mask_dict):
    """
    from volpy dict representation of mask, create standard rxc mask 

    params:
        shape (rxc) image.shape
        mask_dict (volpy representation of mask that includes 'all_points_x' and 'all_points_y' keys)
    returns:
        rxc Boolean mask
    """
    y_points = mask_dict['all_points_y']
    x_points = mask_dict['all_points_x']
    mask_coords = np.stack([y_points, x_points]).T
    return polygon2mask(shape, mask_coords)

def box_area(bbox):
    """ 
    return sa of bbox (bbox in xmin, ymin, xmax, ymax form)
    """
    return (bbox[3]-bbox[1])*(bbox[2]-bbox[0])

def normalize_image(image):
    """
    normalize grayscale image to values between 0 and 1 and make sure it is float32
    """
    image_shifted = image - image.min()
    image_normed = image_shifted/image_shifted.max()
    return np.array(image_normed, dtype=np.float32)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    from mrcnn
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color=(1,0,0), alpha=0.5):
    """
    Apply the given mask to the image. Alpha is opacity, from 0 (transparent) to 1 (opaque)

    From mrcnn
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def apply_masks(image, data_masks, color=(1,0, 0), alpha=0.5):
    """
    apply many masks (N x H x W) to given image
    adapted from mrcnn
    """
    masked_image = image.copy()
    
    for mask_ind, mask in enumerate(data_masks):
        # print(mask_ind)
        masked_image = apply_mask(masked_image, mask, color, alpha=alpha)
        
    return masked_image

def collate_fn(batch):
    """
    from torchvision utils.py
    """
    return tuple(zip(*batch))