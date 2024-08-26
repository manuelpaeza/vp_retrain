#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

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
        ax = plt.gca() #figure this out
        
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

def draw_bboxes(bboxes, color='white', ax=None, line_width=0.5):
    """
    given Nx4 bounding boxes, draw them all on given axes object

    Returns axes object and list of rects
    """
    if ax is None:
        ax = plt.gca() #figure this out

    num_boxes = len(bboxes)
    all_rects = []
    for bbox in bboxes:
        ax, rect = draw_bbox(bbox, color=color, ax=ax, line_width=line_width)
        all_rects.append(rect)
        
    return ax, all_rects
