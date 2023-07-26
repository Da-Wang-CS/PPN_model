#! /usr/bin/env python
# Carlos X. Soto, csoto@bnl.gov, 2022


import os
import time
import math
import argparse
from easydict import EasyDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BarValueExtractor import *
from model.Datasets import *
from model.dataloaders import get_dataloaders
from model.nms import *
from model.utils.metrics import *
from model.utils.map_conversions import * 

## Evaluate performance on test sets

if len(sys.argv) != 3:
    print("USAGE: <checkpoint_dir> <eval_dataloader>")
    exit(1)

checkpoint_dir = sys.argv[1]
dataset_id = sys.argv[2]

#checkpoint_dir = 'ppn_checkpoints/20211002_ml_origin'
#checkpoint_dir = 'ppn_checkpoints/20211107_synth_ptsloss_1500'
#checkpoint_dir = 'ppn_checkpoints/20211107_synth_align_1600'
#checkpoint_dir = 'ppn_checkpoints/20211108_annot_3k-5k'
#checkpoint_dir = 'ppn_checkpoints/20211003_synthetic'
#checkpoint_dir = 'ppn_checkpoints'
#checkpoint_dir = 'ppn_checkpoints/submission_chks'

#eval_dataloader = val_aug_dataloader
#eval_dataloader = test4_dataloader
#eval_dataloader = zhao_test_dl

#eval_dataloader = get_dataloaders(dataset_id, 100)
eval_dataloader, _= get_dataloaders(dataset_id, 5)

# defaults
#epoch = 9800
epoch = 7
pnt_detect_thresh = 0.9
cls_conf_thresh = 0.75
eval_thresh = 1.5 / 56


print('epoch, det_thresh, cls_thresh, ev_thresh, time, bar P, bar R, tick P, tick R, errorup P, errorup R, errordown P, errordown R, bar F1, tick F1, errorup F1, errordown F1, mean F1')
#for epoch in range(1000,1501,20):
#for epoch in [1600,9800,300]:
#for pnt_detect_thresh in range(80,100):
#    pnt_detect_thresh = pnt_detect_thresh / 100
#for cls_conf_thresh in range(50,100,5):
#    cls_conf_thresh = cls_conf_thresh / 100
#for eval_thresh in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
for eval_thresh in [2.8, 0.5]:
#    eval_thresh = eval_thresh / 56
    
    #checkpoint_name = f'ppn_chk_epoch_{epoch:04}.pth'
    checkpoint_name = f'ppn_chk_epoch_0001.pth' ##### TEST #####
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    model = BarValueExtractor()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tstart = time.time()
    
    img_metrics = []   # (precision, recall, F1) x2 (bar, tick)

    #for k, (img_path, img, targets) in enumerate(eval_dataloader):

    for (img_path, img, targets) in tqdm(eval_dataloader):
        # gt_orient, gt_origin, gt_cls_map, gt_reg_map, _, _ = targets

        #img_path = targets[0]
        #img = targets[1]
        #targets = targets[2]

        print("errorup: ", targets[3])
        print("errordown: ", targets[4])
        
        gt_orient, gt_origin, gt_cls_map, gt_reg_map = targets
        
        gt_bars, gt_ticks, gt_errorup, gt_errordown = pts_map_to_lists_v2(gt_cls_map, gt_reg_map)

        orient_pred, origin_pred, pred_cls_map, pred_reg_map = model(img)
              
#       bars, ticks = get_pred_bars_ticks(pred_cls_map, pred_reg_map, pt_thresh = 0.99, conf_thresh = 0.7)
        bars, ticks, errorup, errordown = get_pred_bars_ticks(pred_cls_map, pred_reg_map, pnt_detect_thresh, cls_conf_thresh)
        bars, ticks, errorup, errordown = nms(bars, ticks, errorup, errordown)

        # EVALUATION per image
        for i, path in enumerate(img_path):
            barP, barR, tickP, tickR, euP, euR, edP, edR = evaluate_pts(gt_bars[i], gt_ticks[i], gt_errorup[i], gt_errordown[i], bars[i], ticks[i], errorup[i], errordown[i], eval_thresh)
            barF1, tickF1, euF1, edF1 = f1(barP, barR), f1(tickP, tickR), f1(euP, euR), f1(edP, edR)
            img_metrics.append((barP, barR, barF1, tickP, tickR, tickF1, euP, euR, euF1, edP, edR, edF1))

    # mean of bar and tick metrics (precision, recall, F1) @ threshold of 1.5/56
    mbP = sum([im[0] for im in img_metrics]) / len(img_metrics)
    mbR = sum([im[1] for im in img_metrics]) / len(img_metrics)
    mbF1 = sum([im[2] for im in img_metrics]) / len(img_metrics)
    mtP = sum([im[3] for im in img_metrics]) / len(img_metrics)
    mtR = sum([im[4] for im in img_metrics]) / len(img_metrics)
    mtF1 = sum([im[5] for im in img_metrics]) / len(img_metrics)
    meuP = sum([im[6] for im in img_metrics]) / len(img_metrics)
    meuR = sum([im[7] for im in img_metrics]) / len(img_metrics)
    meuF1 = sum([im[8] for im in img_metrics]) / len(img_metrics)
    medP = sum([im[9] for im in img_metrics]) / len(img_metrics)
    medR = sum([im[10] for im in img_metrics]) / len(img_metrics)
    medF1 = sum([im[11] for im in img_metrics]) / len(img_metrics)
    avgF1 = (mbF1 + mtF1 + meuF1 + medF1) / 4.

    print(f'{epoch}, {pnt_detect_thresh:.2f}, {cls_conf_thresh:.2f}, ' +
          f'{eval_thresh:.1f}, {time.time() - tstart:.1f}, '+
          f'{mbP:.4f}, {mbR:.4f}, {mtP:.4f}, {mtR:.4f}, {meuP:.4f}, {meuR:.4f}, {medP:.4f}, {medR:.4f}, '+
          f'{mbF1:.4f}, {mtF1:.4f}, {meuF1:.4f}, {medF1:.4f}, {avgF1:.6f}')