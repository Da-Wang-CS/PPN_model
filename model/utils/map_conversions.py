# Carlos X. Soto, csoto@bnl.gov, 2022

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("agg")

id_to_name = {0: 'None', 1: 'bar', 2: 'tick', 3: 'errorup', 4: 'errordown'}
name_to_id = {'None': 0, 'bar': 1, 'tick': 2, 'errorup': 3, 'errordown': 4}

def score_points_list(gt_bars, gt_ticks, pred_bars, pred_ticks):
    pass

def vis_cls_map(gt_cls_map, pred_cls_map):
    """
    plt.imshow(cls_map, cmap = "autumn")
    plt.colorbar()
            
    plt.savefig("cls_map_heat.png", format = "png")
    """

    pred_types = ["Background", "Bar", "Tick", "Upper Error", "Lower Error"]

    plt.rcParams["figure.figsize"] = (15,15)

    plt.subplot(3, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(gt_cls_map[0])
    plt.colorbar()

    for idx, label in enumerate(pred_types):
        plt.subplot(3, 2, idx + 2)
        plt.title(f"{label} Prediction")
        if idx != 0:
            plt.imshow(F.sigmoid(pred_cls_map.detach()[0][idx]))
        else:
            plt.imshow(F.sigmoid(pred_cls_map.detach()[0][0]) * -1.)
        
        plt.colorbar()

    """
    plt.subplot(3, 2, 2)
    plt.imshow(F.sigmoid(pred_cls_map.detach()[0][0] * -1.))
    plt.subplot(3, 2, 3)
    plt.imshow(F.sigmoid(pred_cls_map.detach()[0][1]))
    plt.subplot(3, 2, 4)
    plt.imshow(F.sigmoid(pred_cls_map.detach()[0][2]))
    plt.subplot(3, 2, 5)
    plt.imshow(F.sigmoid(pred_cls_map.detach()[0][3]))
    plt.subplot(3, 2, 6)
    plt.imshow(F.sigmoid(pred_cls_map.detach()[0][4]))
    plt.colorbar()
    """
    
    plt.savefig("cls_map_heat.png", format = "png")

    plt.close("all")

# convert list of points of different classes (in single image)
# to (56x56) map of point classes and regression values
def pts_lists_to_map(pts_lists, imw, imh, pad, mapsize = 56.0):
    
    batch_size = len(pts_lists)
    cls_map = torch.zeros((56, 56), dtype = int)
    reg_map = torch.zeros((2, 56, 56))
    
    # pts_lists is list of lists of points
    # one top-level list per class
    # first class gets id 1 (id 0 is None class)
    
    for cls_id, cls_list in enumerate(pts_lists):
        for point in cls_list:
            
            #print('pad', pad)
            #print('point', point)
            
            # (x, y) pairs in range of image coords
            x = (point[0] + pad[0]) / (imw + pad[0] + pad[1])
            y = (point[1] + pad[2]) / (imh + pad[2] + pad[3])
            
            # compute map coordinates and regression values
            posx = math.floor(x * mapsize)
            posy = math.floor(y * mapsize)
            regx = x * mapsize - posx - 0.5
            regy = y * mapsize - posy - 0.5
            
            #cls_map[posx, posy] = cls_id + 1
            #reg_map[0, posx, posy] = regx
            #reg_map[1, posx, posy] = regy
            # transpose maps
            cls_map[posy, posx] = cls_id + 1
            reg_map[0, posy, posx] = regx
            reg_map[1, posy, posx] = regy            
            
            #cls_map = cls_map.transpose(0,1)
            #reg_map = reg_map.transpose(1,2)
    
    return cls_map, reg_map

# version for ground truth maps
def pts_map_to_lists_v2(pts_cls_map, pts_reg_map):
    # seperate lists per batch image
    bars = [[] for im in range(pts_cls_map.shape[0])]
    ticks = [[] for im in range(pts_reg_map.shape[0])]
    errorup = [[] for im in range(pts_cls_map.shape[0])]
    errordown = [[] for im in range(pts_cls_map.shape[0])]

    #classes = torch.argmax(pts_cls_map, 1)
    classes = pts_cls_map
    pts_im, pts_x, pts_y = torch.nonzero(classes, as_tuple=True)
    
    for im, x, y in zip(pts_im, pts_x, pts_y):
        # position of point [0-1] (midpoint of pixel)
        cls = classes[im, x, y]
        pos_x = (x.float() * 2 + 1) / (classes.shape[1] * 2)
        pos_y = (y.float() * 2 + 1) / (classes.shape[2] * 2)
        #pos_x = x.float() / classes.shape[1]
        #pos_y = y.float() / classes.shape[2]
        
        # offset from midpoint of this pixel on feature map
        reg = pts_reg_map[im, :, x, y]
        pos_x += reg[0] / classes.shape[1]
        pos_y += reg[1] / classes.shape[1]
        
        # assign to class lists
        if cls == name_to_id['bar']:
            bars[im].append((pos_x, pos_y))
        elif cls == name_to_id['tick']:
            ticks[im].append((pos_x, pos_y))
        elif cls == name_to_id['errorup']:
            errorup[im].append((pos_x, pos_y))
        elif cls == name_to_id['errordown']:
            errordown[im].append((pos_x, pos_y))
    
    return bars, ticks, errorup, errordown





#####!!!!! NOTE: DEPRECATED FUNCTIONS BELOW !!!!!#####





# convert (56x56) map of point classes and regression values
# to list of points of each class
# NOTE: CLASSES ARE NOT SOFT-MAXED PROBABILITIES
# NOTE: NEED TO MAKE REGRESSION BOUND TO [-1,1] (scaled to one pixel in 56x56 map)
def pts_map_to_lists(pts_cls_pred, pts_reg_pred):
    raise DeprecationWarning("Please use pts_map_to_lists_v2 instead")
    # seperate lists per batch image
    bars = [[] for im in range(pts_cls_pred.shape[0])]
    ticks = [[] for im in range(pts_cls_pred.shape[0])]
    
    #print('pts_cls_pred.shape', pts_cls_pred.shape)
    #print(pts_cls_pred[0,:,:5,:5])
    #return
    
    # TODO: use a threshold for non-empty point, rather than simple argmax
    
    classes = torch.argmax(pts_cls_pred, 1)
    pts_im, pts_x, pts_y = torch.nonzero(classes, as_tuple=True)
    
    for im, x, y in zip(pts_im, pts_x, pts_y):
        # position of point [0-1] (midpoint of pixel)
        cls = classes[im, x, y]
        pos_x = (x.float() * 2 + 1) / (classes.shape[1] * 2)
        pos_y = (y.float() * 2 + 1) / (classes.shape[2] * 2)
        
        # offset from midpoint of this pixel on feature map
        reg = pts_reg_pred[im, :, x, y]
        pos_x += reg[0] / classes.shape[1]
        pos_y += reg[1] / classes.shape[1]
        
        # assign to class lists
        if cls == name_to_id['bar']:
            bars[im].append((pos_x, pos_y))
        elif cls == name_to_id['tick']:
            ticks[im].append((pos_x, pos_y))
    
    return bars, ticks