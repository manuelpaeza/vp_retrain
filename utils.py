"""
utils.py

Updated: Manuel Paez, Changjia Cai
Date: January 8, 2025
"""

import colorsys
import cv2
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy
from scipy.ndimage import label, gaussian_filter
from scipy.optimize import linear_sum_assignment
import shutil
from skimage.draw import polygon, polygon2mask
from skimage.filters import sobel
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, closing
from skimage.segmentation import watershed
import tempfile
import time
import torch 
from torchvision.transforms.v2 import functional as F
from typing import Any, Optional
import zipfile

def distance_masks(M_s:list, cm_s: list[list], max_dist: float, enclosed_thr: Optional[float] = None) -> list:
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order,
    with matrix i compared with matrix i+1

    Args:
        M_s: tuples of 1-D arrays
            The thresholded A matrices (masks) to compare, output of threshold_components

        cm_s: list of list of 2-ples
            the centroids of the components in each M_s

        max_dist: float
            maximum distance among centroids allowed between components. This corresponds to a distance
            at which two components are surely disjoined

        enclosed_thr: float
            if not None set distance to at most the specified value when ground truth is a subset of inferred

    Returns:
        D_s: list of matrix distances

    Raises:
        Exception: 'Nan value produced. Error in inputs'

    """
    D_s = []

    for gt_comp, test_comp, cmgt_comp, cmtest_comp in zip(M_s[:-1], M_s[1:], cm_s[:-1], cm_s[1:]):

        # todo : better with a function that calls itself
        # not to interfere with M_s
        gt_comp = gt_comp.copy()[:, :]
        test_comp = test_comp.copy()[:, :]

        # the number of components for each
        nb_gt = np.shape(gt_comp)[-1]
        nb_test = np.shape(test_comp)[-1]
        D = np.ones((nb_gt, nb_test))

        cmgt_comp = np.array(cmgt_comp)
        cmtest_comp = np.array(cmtest_comp)
        if enclosed_thr is not None:
            gt_val = gt_comp.T.dot(gt_comp).diagonal()
        for i in range(nb_gt):
            # for each components of gt
            k = gt_comp[:, np.repeat(i, nb_test)] + test_comp
            # k is correlation matrix of this neuron to every other of the test
            for j in range(nb_test):   # for each components on the tests
                dist = np.linalg.norm(cmgt_comp[i] - cmtest_comp[j])
                                       # we compute the distance of this one to the other ones
                if dist < max_dist:
                                       # union matrix of the i-th neuron to the jth one
                    union = k[:, j].sum()
                                       # we could have used OR for union and AND for intersection while converting
                                       # the matrice into real boolean before

                    # product of the two elements' matrices
                    # we multiply the boolean values from the jth omponent to the ith
                    intersection = np.array(gt_comp[:, i].T.dot(test_comp[:, j]).todense()).squeeze()

                    # if we don't have even a union this is pointless
                    if union > 0:

                        # intersection is removed from union since union contains twice the overlapping area
                        # having the values in this format 0-1 is helpful for the hungarian algorithm that follows
                        D[i, j] = 1 - 1. * intersection / \
                            (union - intersection)
                        if enclosed_thr is not None:
                            if intersection == gt_val[j] or intersection == gt_val[i]:
                                D[i, j] = min(D[i, j], 0.5)
                    else:
                        D[i, j] = 1.

                    if np.isnan(D[i, j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i, j] = 1

        D_s.append(D)
    return D_s


def find_matches(D_s, print_assignment: bool = False) -> tuple[list, list]:
    # todo todocument

    matches = []
    costs = []
    t_start = time.time()
    for ii, D in enumerate(D_s):
        # we make a copy not to set changes in the original
        DD = D.copy()
        if np.sum(np.where(np.isnan(DD))) > 0:
            logging.error('Exception: Distance Matrix contains invalid value NaN')
            raise Exception('Distance Matrix contains invalid value NaN')

        # we do the hungarian
        indexes = linear_sum_assignment(DD)
        indexes2 = [(ind1, ind2) for ind1, ind2 in zip(indexes[0], indexes[1])]
        matches.append(indexes)
        DD = D.copy()
        total = []
        # we want to extract those information from the hungarian algo
        for row, column in indexes2:
            value = DD[row, column]
            if print_assignment:
                logging.debug(('(%d, %d) -> %f' % (row, column, value)))
            total.append(value)
        logging.debug(('FOV: %d, shape: %d,%d total cost: %f' % (ii, DD.shape[0], DD.shape[1], np.sum(total))))
        logging.debug((time.time() - t_start))
        costs.append(total)
        # send back the results in the format we want
    return matches, costs



def nf_match_neurons_in_binary_masks(masks_gt,
                                     masks_comp,
                                     thresh_cost=.7,
                                     min_dist=10,
                                     print_assignment=False,
                                     plot_results=False,
                                     Cn=None,
                                     labels=['Session 1', 'Session 2'],
                                     cmap='gray',
                                     D=None,
                                     enclosed_thr=None,
                                     colors=['red', 'white']):
    """
    Match neurons expressed as binary masks. Uses Hungarian matching algorithm

    Args:
        masks_gt: bool ndarray  components x d1 x d2
            ground truth masks

        masks_comp: bool ndarray  components x d1 x d2
            mask to compare to

        thresh_cost: double
            max cost accepted

        min_dist: min distance between cm

        print_assignment:
            for hungarian algorithm

        plot_results: bool

        Cn:
            correlation image or median

        D: list of ndarrays
            list of distances matrices

        enclosed_thr: float
            if not None set distance to at most the specified value when ground truth is a subset of inferred

    Returns:
        idx_tp_1:
            indices true pos ground truth mask

        idx_tp_2:
            indices true pos comp

        idx_fn_1:
            indices false neg

        idx_fp_2:
            indices false pos

    """

    _, d1, d2 = np.shape(masks_gt)
    dims = d1, d2

    # transpose to have a sparse list of components, then reshaping it to have a 1D matrix red in the Fortran style
    A_ben = scipy.sparse.csc_matrix(np.reshape(masks_gt[:].transpose([1, 2, 0]), (
        np.prod(dims),
        -1,
    ), order='F'))
    A_cnmf = scipy.sparse.csc_matrix(np.reshape(masks_comp[:].transpose([1, 2, 0]), (
        np.prod(dims),
        -1,
    ), order='F'))

    # have the center of mass of each element of the two masks
    cm_ben  = [scipy.ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [scipy.ndimage.center_of_mass(mm) for mm in masks_comp]

    if D is None:
        # find distances and matches
        # find the distance between each masks
        D = distance_masks([A_ben, A_cnmf], [cm_ben, cm_cnmf], min_dist, enclosed_thr=enclosed_thr)

    level = 0.98

    matches, costs = find_matches(D, print_assignment=print_assignment)
    matches = matches[0]
    costs = costs[0]

    # compute precision and recall
    TP = np.sum(np.array(costs) < thresh_cost) * 1.
    FN = np.shape(masks_gt)[0] - TP
    FP = np.shape(masks_comp)[0] - TP
    TN = 0

    performance = dict()
    performance['recall'] = TP / (TP + FN)
    performance['precision'] = TP / (TP + FP)
    performance['accuracy'] = (TP + TN) / (TP + FP + FN + TN)
    performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)
    logging.debug(performance)

    idx_tp = np.where(np.array(costs) < thresh_cost)[0]
    idx_tp_ben = matches[0][idx_tp]    # ground truth
    idx_tp_cnmf = matches[1][idx_tp]   # algorithm - comp

    idx_fn = np.setdiff1d(list(range(np.shape(masks_gt)[0])), matches[0][idx_tp])

    idx_fp = np.setdiff1d(list(range(np.shape(masks_comp)[0])), matches[1][idx_tp])

    idx_fp_cnmf = idx_fp

    idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp = idx_tp_ben, idx_tp_cnmf, idx_fn, idx_fp_cnmf

    if plot_results:
        #try:   # Plotting function
        plt.rcParams['pdf.fonttype'] = 42
        #font = {'family': 'Myriad Pro', 'weight': 'regular', 'size': 10}
        #pl.rc('font', **font)
        lp, hp = np.nanpercentile(Cn, [5, 95])
        ses_1 = matplotlib.patches.Patch(color=colors[0], label=labels[0])
        ses_2 = matplotlib.patches.Patch(color=colors[1], label=labels[1])
        plt.subplot(1, 2, 1)
        plt.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        #import pdb
        #pdb.set_trace()
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[1], linewidths=1) for mm in masks_comp[idx_tp_comp]]
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[0], linewidths=1) for mm in masks_gt[idx_tp_gt]]
        if labels is None:
            plt.title('MATCHES')
        else:
            plt.title('MATCHES: ' + labels[1] + f'({colors[1][0]}), ' + labels[0] + f'({colors[0][0]})')
        plt.legend(handles=[ses_1, ses_2])
        #pl.show()
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        
        
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[1], linewidths=1) for mm in masks_comp[idx_fp_comp]]
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[0], linewidths=1) for mm in masks_gt[idx_fn_gt]]
        if labels is None:
            plt.title(f'FALSE POSITIVE ({colors[1][0]}), FALSE NEGATIVE ({colors[0][0]})')
        else:
            plt.title(labels[1] + f'({colors[1][0]}), ' + labels[0] + f'({colors[0][0]})')
        #pl.legend(handles=[ses_1, ses_2])
        plt.axis('off')
        plt.show()
        plt.tight_layout()
        #except Exception as e:
        #    logging.warning("not able to plot precision recall: graphics failure")
        #    logging.warning(e)
    return idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance

def norm_nrg(a_):
    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')

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
    
def plot_volpy_segs(image, masks, min_v, max_v, outline_color, outline_width, figsize=(6,10), title=None):
    """
    plot volpy mask outlines

    image from volpy is mean, mean, corr
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=figsize, sharex=True, sharey=True)  # w/h

    # Mean
    ax1.imshow(image[:,:,1], cmap='gray', 
               vmin=np.percentile(image[:,:,1], min_v), 
               vmax=np.percentile(image[:,:,1], max_v));
    ax1.set_title('Mean Image')
    ax2.imshow(image[:,:,1], cmap='gray', 
               vmin=np.percentile(image[:,:,1], min_v), 
               vmax=np.percentile(image[:,:,1], max_v));
    for mask in masks:
        ax2.plot(mask['all_points_x'], 
                 mask['all_points_y'], 
                 color=outline_color, 
                 linewidth=outline_width);
    ax2.set_title('Mean Image Seg')
    
    # Corr
    ax3.imshow(image[:,:,2], cmap='gray', 
               vmin=np.percentile(image[:,:,2], min_v), 
               vmax=np.percentile(image[:,:,2], max_v));
    ax3.set_title('Corr Image')
    ax4.imshow(image[:,:,2], cmap='gray', 
               vmin=np.percentile(image[:,:,2], min_v), 
               vmax=np.percentile(image[:,:,2], max_v));
    for mask in masks:
        ax4.plot(mask['all_points_x'], 
                 mask['all_points_y'], 
                 color=outline_color, 
                 linewidth=outline_width);
    ax4.set_title('Corr Image Seg')

    if title is not None:
        plt.suptitle(title, y=0.99, fontsize=16);
        
    plt.tight_layout()
    
def draw_bbox(bbox, color='white', ax=None, line_width=0.5):
    """
    Draw a single rectangular bounding box on given axes object.
    
    Args:
        bbox: xmin, ymin, xmax, ymax
        color: matplotlib color
        alpha : float opaqueness level (0. to 1., where 1 is opaque), default 0.2
        ax : pyplot.Axes object axes object upon which rectangle will be drawn, default None
    
    Returns:
        ax: pyplot.Axes object
        rect: matplotlib Rectangle object
    """
    from matplotlib.patches import Rectangle
    
    if ax is None:
        ax = plt.gca()
        
    box_origin = (bbox[0], bbox[1])
    box_height = bbox[3] - bbox[1] 
    box_width = bbox[2] - bbox[0]
    # print(box_origin, box_height, box_width)

    rect = Rectangle(box_origin, 
                     width=box_width, 
                     height=box_height,
                     color=color, 
                     alpha=1,
                     fill=None,
                     linewidth=line_width)
    ax.add_patch(rect)

    return ax, rect
    
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
    
def draw_bbox(bbox, color='white', ax=None, line_width=0.5):
    """
    Draw a single rectangular bounding box on given axes object.
    
    Args:
        bbox: xmin, ymin, xmax, ymax
        color: matplotlib color
        alpha : float opaqueness level (0. to 1., where 1 is opaque), default 0.2
        ax : pyplot.Axes object axes object upon which rectangle will be drawn, default None
    
    Returns:
        ax: pyplot.Axes object
        rect: matplotlib Rectangle object
    """
    from matplotlib.patches import Rectangle
    
    if ax is None:
        ax = plt.gca()
        
    box_origin = (bbox[0], bbox[1])
    box_height = bbox[3] - bbox[1] 
    box_width = bbox[2] - bbox[0]
    # print(box_origin, box_height, box_width)

    rect = Rectangle(box_origin, 
                     width=box_width, 
                     height=box_height,
                     color=color, 
                     alpha=1,
                     fill=None,
                     linewidth=line_width)
    ax.add_patch(rect)

    return ax, rect
    
def collate_fn(batch):
    """
    from torchvision utils.py
    """
    return tuple(zip(*batch))
    
def draw_bboxes(bboxes, color='white', ax=None, line_width=0.5):
    """
    given Nx4 bounding boxes, draw them all on given axes object

    Returns axes object and list of rects
    """
    if ax is None:
        ax = plt.gca()

    num_boxes = len(bboxes)
    all_rects = []
    for bbox in bboxes:
        ax, rect = draw_bbox(bbox, color=color, ax=ax, line_width=line_width)
        all_rects.append(rect)
        
    return ax, all_rects

def thresholded_predictions(pred, threshold=0.7):
    """
    get masks and boxes for those above threshold
    """
    numels = len(torch.where(pred['scores'] >= threshold)[0])
    masks = pred['masks'][:numels].squeeze()
    boxes = pred['boxes'][:numels]
    
    return masks, boxes 