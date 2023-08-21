# Carlos X. Soto, csoto@bnl.gov, 2022

#import numpy as np

def count_nonzero(arr):
    ct = 0

    print(arr)

    for a in range(len(arr)):
        for b in range(len(arr[a])):
            if arr[a][b] != 0:
                ct += 1

    return ct

"""
def old_nonzero(arr):
    ct = 0

    for a in range(len(arr)):
        if arr[a] != 0:
            ct += 1

    return ct
"""
    
# evaluation function (for a single image)
def evaluate_pts(gt_bars, gt_ticks, gt_errorup, gt_errordown, pred_bars, pred_ticks, pred_errorup, pred_errordown, dist_thresh = 1.5 / 56):
    #for gb in gt_bars:
    #    for pb in pred_bars:
    #        pixel_dist = ((gb[0] - pb[0]) ** 2 + (gb[1] - pb[1]) ** 2) ** (0.5)
    
    #print("gt_bars: ", gt_bars)
    #print("pred_bars", pred_bars)

    #print(gt_ticks)

    # fill a matching matrix, check rows (ground truth) for uniqueness
    bar_matches = [[((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) if
                    ((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in pred_bars] for gb in gt_bars]
    for i in range(len(gt_bars)):
        min_dist = min([bm for bm in bar_matches[i] if bm > 0], default = 0.)
        bar_matches[i] = [m if m <= min_dist else 0 for m in bar_matches[i]]

    #num_matches = np.count_nonzero(bar_matches) ## ERROR: numpy must be in cpu
    num_matches = count_nonzero(bar_matches)

    """
    print("bar matches: ", bar_matches)
    print("pred: ", len(pred_bars))
    print("gt: ", len(gt_bars))
    """
    #print(gt_bars)
    #print(pred_bars)

    bar_precision = (num_matches / (len(pred_bars) * 2)) if len(pred_bars) != 0 else 0
    bar_recall = (num_matches / (len(gt_bars) * 2)) if len(gt_bars) != 0 else 0
    
    # errorup
    eu_matches = [[((geu[1] - peu[1]) ** 2 + (geu[0] - peu[0]) ** 2) ** (0.5) if
                   ((geu[1] - peu[1]) ** 2 + (geu[0] - peu[0]) ** 2) ** (0.5) < dist_thresh
                   else 0
                   for peu in pred_errorup] for geu in gt_errorup]
    for i in range(len(gt_errorup)):
        min_dist = min([eum for eum in eu_matches[i] if eum > 0], default = 0.)
        eu_matches[i] = [m if m <= min_dist else 0 for m in eu_matches[i]]
    
    num_matches = count_nonzero(eu_matches)

    eu_precision = (num_matches / (len(pred_errorup) * 2)) if len(pred_errorup) != 0 else 0
    eu_recall = (num_matches / (len(gt_errorup) * 2)) if len(gt_errorup) != 0 else 0

    
    # errordown
    ed_matches = [[((ged[1] - ped[1]) ** 2 + (ged[0] - ped[0]) ** 2) ** (0.5) if
                   ((ged[1] - ped[1]) ** 2 + (ged[0] - ped[0]) ** 2) ** (0.5) < dist_thresh
                   else 0
                   for ped in pred_errordown] for ged in gt_errordown]
    for i in range(len(gt_errordown)):
        min_dist = min([edm for edm in ed_matches[i] if edm > 0], default = 0.)
        ed_matches[i] = [m if m <= min_dist else 0 for m in ed_matches[i]]
    
    num_matches = count_nonzero(ed_matches)
    
    ed_precision = (num_matches / (len(pred_errordown) * 2)) if len(pred_errordown) != 0 else 0
    ed_recall = (num_matches / (len(gt_errordown) * 2)) if len(gt_errordown) != 0 else 0
    
    
    ## again for ticks...
    tick_matches = [[((gt[1] - pt[1]) ** 2 + (gt[0] - pt[0]) ** 2) ** (0.5) if
                     ((gt[1] - pt[1]) ** 2 + (gt[0] - pt[0]) ** 2) ** (0.5) < dist_thresh
                     else 0
                     for pt in pred_ticks] for gt in gt_ticks]
    for i in range(len(gt_ticks)):
        min_dist = min([tm for tm in tick_matches[i] if tm > 0], default = 0.)
        tick_matches[i] = [m if m <= min_dist else 0 for m in tick_matches[i]]
    
    #num_matches = np.count_nonzero(tick_matches) ## NOTE: numpy must be in cpu
    num_matches = count_nonzero(tick_matches)
    
    tick_precision = (num_matches / (len(pred_ticks) * 2)) if len(pred_ticks) != 0 else 0
    tick_recall = (num_matches / (len(gt_ticks) * 2)) if len(gt_ticks) != 0 else 0
    
    return bar_precision, bar_recall, tick_precision, tick_recall, eu_precision, eu_recall, ed_precision, ed_recall

def evaluate_pts_err(gt_bars, gt_ticks, gt_errorup, gt_errordown, pred_bars, pred_ticks, pred_errorup, pred_errordown, dist_thresh = 1.5 / 56):
    # fill a matching matrix, check rows (ground truth) for uniqueness
    bar_matches = [[((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) if
                    ((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in pred_bars] for gb in gt_bars]
    min_bar_dists = [min([bm for bm in row if bm > 0], default = 0.) for row in bar_matches]

    eu_matches = [[((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) if
                    ((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in pred_errorup] for gb in gt_errorup]
    min_eu_dists = [min([bm for bm in row if bm > 0], default = 0.) for row in eu_matches]

    ed_matches = [[((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) if
                    ((gb[1] - pb[1]) ** 2 + (gb[0] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in pred_errordown] for gb in gt_errordown]
    min_ed_dists = [min([bm for bm in row if bm > 0], default = 0.) for row in ed_matches]
    
    tick_matches = [[((gt[1] - pt[1]) ** 2 + (gt[0] - pt[0]) ** 2) ** (0.5) if
                     ((gt[1] - pt[1]) ** 2 + (gt[0] - pt[0]) ** 2) ** (0.5) < dist_thresh
                     else 0
                     for pt in pred_ticks] for gt in gt_ticks]
    min_tick_dists = [min([tm for tm in row if tm > 0], default = 0.) for row in tick_matches]
    
    return sum(min_bar_dists) + sum(min_tick_dists) + sum(min_eu_dists) + sum(min_ed_dists)

def f1(precision, recall):
    return (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.