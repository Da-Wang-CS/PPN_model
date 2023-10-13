# Carlos X. Soto, csoto@bnl.gov, 2022

import torch

def getPredPts(pred_cls_map, pred_reg_map, none_cls_map, pt_thresh = 0.8, conf_thresh = 0.5):
    points = []
    
    pts_mask = torch.sigmoid(none_cls_map).lt(pt_thresh)

    masked_pts = (torch.sigmoid(pred_cls_map) * pts_mask).gt(conf_thresh)
    b_im, b_x, b_y = torch.nonzero(masked_pts, as_tuple=True)
    for im, x, y in zip(b_im, b_x, b_y):
        pos_x = (x.float() * 2 + 1) / (pts_mask.shape[1] * 2)    # MIDDLE of each point on 56x56 grid
        pos_y = (y.float() * 2 + 1) / (pts_mask.shape[2] * 2)
        reg = pred_reg_map[im, :, x, y]
        pos_x += reg[0] / (pts_mask.shape[1] * 2)                # reg[0] corresponds to y, and vice-versa
        pos_y += reg[1] / (pts_mask.shape[2] * 2)
        conf = torch.sigmoid(pred_cls_map[im, 1, x, y])
        points[im].append((pos_x, pos_y, conf))                    # x/y flipped

    return points

# expect cls_map.shape == [N,3,56,56]
def get_pred_bars_ticks(pred_cls_map, pred_reg_map, pt_thresh = 0.8, conf_thresh = 0.5):

    # raise DeprecationWarning("Deprecated for the parallel PPN model")
    #print("reg:", pred_reg_map.shape)

    # seperate lists per batch image
    bars = [[] for im in range(pred_cls_map.shape[0])]
    ticks = [[] for im in range(pred_cls_map.shape[0])]
    errorup = [[] for im in range(pred_cls_map.shape[0])]
    errordown = [[] for im in range(pred_cls_map.shape[0])]
    
    # map of non-background points
    pts_mask = torch.sigmoid(pred_cls_map[:,0]).lt(pt_thresh)
    
    masked_bars = (torch.sigmoid(pred_cls_map[:,1]) * pts_mask).gt(conf_thresh)
    b_im, b_x, b_y = torch.nonzero(masked_bars, as_tuple=True)
    for im, x, y in zip(b_im, b_x, b_y):
        pos_x = (x.float() * 2 + 1) / (pts_mask.shape[1] * 2)    # MIDDLE of each point on 56x56 grid
        pos_y = (y.float() * 2 + 1) / (pts_mask.shape[2] * 2)
        reg = pred_reg_map[im, 1, :, x, y]
        pos_x += reg[0] / (pts_mask.shape[1] * 2)                # reg[0] corresponds to y, and vice-versa
        pos_y += reg[1] / (pts_mask.shape[2] * 2)
        conf = torch.sigmoid(pred_cls_map[im, 1, 1, x, y])
        bars[im].append((pos_x, pos_y, conf))                    # x/y flipped

    masked_ticks = (torch.sigmoid(pred_cls_map[:,2]) * pts_mask).gt(conf_thresh)
    t_im, t_x, t_y = torch.nonzero(masked_ticks, as_tuple=True)
    for im, x, y in zip(t_im, t_x, t_y):
        pos_x = (x.float() * 2 + 1) / (pts_mask.shape[1] * 2)
        pos_y = (y.float() * 2 + 1) / (pts_mask.shape[2] * 2)
        reg = pred_reg_map[im, 2, :, x, y]
        pos_x += reg[0] / (pts_mask.shape[1] * 2)
        pos_y += reg[1] / (pts_mask.shape[2] * 2)
        conf = torch.sigmoid(pred_cls_map[im, 2, 2, x, y])
        ticks[im].append((pos_x, pos_y, conf))                   # x/y flipped

    masked_errorup = (torch.sigmoid(pred_cls_map[:,3]) * pts_mask).gt(conf_thresh)
    eu_im, eu_x, eu_y = torch.nonzero(masked_errorup, as_tuple = True)
    for im, x, y in zip(eu_im, eu_x, eu_y):
        pos_x = (x.float() * 2 + 1) / (pts_mask.shape[1] * 2)
        pos_y = (y.float() * 2 + 1) / (pts_mask.shape[2] * 2)
        reg = pred_reg_map[im, 3, :, x, y]
        pos_x += reg[0] / (pts_mask.shape[1] * 2)
        pos_y += reg[1] / (pts_mask.shape[2] * 2)
        conf = torch.sigmoid(pred_cls_map[im, 3, 3, x, y])
        errorup[im].append((pos_x, pos_y, conf))

    masked_errordown = (torch.sigmoid(pred_cls_map[:,4]) * pts_mask).gt(conf_thresh)
    ed_im, ed_x, ed_y = torch.nonzero(masked_errordown, as_tuple = True)
    for im, x, y in zip(ed_im, ed_x, ed_y):
        pos_x = (x.float() * 2 + 1) / (pts_mask.shape[1] * 2)
        pos_y = (y.float() * 2 + 1) / (pts_mask.shape[2] * 2)
        reg = pred_reg_map[im, 4, :, x, y]
        pos_x += reg[0] / (pts_mask.shape[1] * 2)
        pos_y += reg[1] / (pts_mask.shape[2] * 2)
        conf = torch.sigmoid(pred_cls_map[im, 4, 4, x, y])
        errordown[im].append((pos_x, pos_y, conf))

    #print("bars: ", bars)

    # non-maximum suppression here... can also apply some heuristic rules
    # (e.g. ticks in a single column, bars evenly spaced horizontally)
    
    return bars, ticks, errorup, errordown

# non-maximum suppression (per image in batch, AFTER model produces cls/reg maps)
def nms(bars, ticks, errorup, errordown, thresh = 1.5 / 56):
    nms_bars = [[] for im in range(len(bars))]
    nms_ticks = [[] for im in range(len(ticks))]
    nms_errorup = [[] for im in range(len(errorup))]
    nms_errordown = [[] for im in range(len(errordown))]
    
    # for bars, make radius anisotropic (5x more wiggle-room left-right)
    for im, im_pts in enumerate(bars):
        sorted_impts = sorted(im_pts, key=lambda p:-p[2])        # sort by conf score
        checked_pts = []                                         # keep track of 'used' points
        for k, pt in enumerate(sorted_impts):
            if k in checked_pts:
                continue
            neighbors_inds = [i for i,p in enumerate(sorted_impts)
                              if abs(p[0]-pt[0]) + abs(p[1]-pt[1]) / 5 < thresh
                              if i not in checked_pts]
            checked_pts.extend(neighbors_inds)
            nms_bars[im].append(pt)
    
    for im, im_pts in enumerate(ticks):
        sorted_impts = sorted(im_pts, key=lambda p:-p[2])        # sort by conf score
        checked_pts = []                                         # keep track of 'used' points
        for k, pt in enumerate(sorted_impts):
            if k in checked_pts:
                continue
            neighbors_inds = [i for i,p in enumerate(sorted_impts)
                              if abs(p[0]-pt[0]) + abs(p[1]-pt[1]) < thresh
                              if i not in checked_pts]
            checked_pts.extend(neighbors_inds)
            nms_ticks[im].append(pt)

    for im, im_pts in enumerate(errorup):
        sorted_impts = sorted(im_pts, key=lambda p:-p[2])        # sort by conf score
        checked_pts = []                                         # keep track of 'used' points
        for k, pt in enumerate(sorted_impts):
            if k in checked_pts:
                continue
            neighbors_inds = [i for i,p in enumerate(sorted_impts)
                              if abs(p[0]-pt[0]) + abs(p[1]-pt[1]) / 5 < thresh
                              if i not in checked_pts]
            checked_pts.extend(neighbors_inds)
            nms_errorup[im].append(pt)

    for im, im_pts in enumerate(errordown):
        sorted_impts = sorted(im_pts, key=lambda p:-p[2])        # sort by conf score
        checked_pts = []                                         # keep track of 'used' points
        for k, pt in enumerate(sorted_impts):
            if k in checked_pts:
                continue
            neighbors_inds = [i for i,p in enumerate(sorted_impts)
                              if abs(p[0]-pt[0]) + abs(p[1]-pt[1]) / 5 < thresh
                              if i not in checked_pts]
            checked_pts.extend(neighbors_inds)
            nms_errordown[im].append(pt)    
    
    return nms_bars, nms_ticks, nms_errorup, nms_errordown