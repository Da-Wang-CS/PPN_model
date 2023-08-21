import torch
import sys
import os
from model.dataloaders import load_test_data
from model.BarValueExtractor import BarValueExtractor
from model.nms import *
from model.utils.metrics import *
from model.utils.map_conversions import *
from model.utils.readeval import *
from tqdm.auto import tqdm
from PIL import Image, ImageDraw

if len(sys.argv) != 4:
    raise Warning("USAGE: <dataset_id> <checkpoint> <vis_dir>")

dataset_id = sys.argv[1]
checkpoint_path = sys.argv[2]
vis_dir = sys.argv[3]
data_dir = f"datasets/{dataset_id}/data/"
img_dir = f"datasets/{dataset_id}/imgs/"

test_dataloder = load_test_data(dataset_id, 1)
model = BarValueExtractor()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

im_metrics = []
eval_thresh = 2.8

def drawFunction(img, orient, bar, tick, eu, ed, pred, color = "blue", r = 3):
    points = [bar, tick, eu, ed]
    predColors = ["red", "green", "yellow", "orange"]

    if round(orient.item()) == 1:
        img.text((10, 10), "Ground truth: horizontal", (0, 255, 0)) if not pred else img.text((10, 20), "Predicted: horizontal", (255, 0, 0))
    elif round(orient.item()) == 0:
        img.text((10, 10), "Ground truth: vertical", (0, 255, 0)) if not pred else img.text((10, 20), "Predicted: vertical", (255, 0, 0))

    for colSel, item in enumerate(points):
        for idx in range(len(item)):
            if not pred:
                img.ellipse([float(item[idx][0] - r), float(item[idx][1] - r), float(item[idx][0] + r), float(item[idx][1] + r)], fill = color, outline = color, width = 4)
            else:
                img.ellipse([float(item[idx][0] - r), float(item[idx][1] - r), float(item[idx][0] + r), float(item[idx][1] + r)], fill = predColors[colSel], outline = predColors[colSel], width = 4)

def scale(ptsList, dim: tuple[2]):
    result = []

    for item in ptsList:
        for b, coord in enumerate(item):
            item[b] = (coord[0] * dim[0], coord[1] * dim[1])
        result.append(item)

    return result

def rmPad(ptsList, pad):
    result = []
    
    for item in ptsList:
        for b, coord in enumerate(item):
            item[b] = (coord[0] - pad[0], coord[1] - pad[2])
        result.append(item)

    return result


outlabel = 0

for (im_path, im, targets, pad) in tqdm(test_dataloder, desc = "Testing..."):
    gt_orient, gt_origin, gt_cls_map, gt_reg_map = targets
    
    gt_bars, gt_ticks, gt_errorup, gt_errordown = pts_map_to_lists_v2(gt_cls_map, gt_reg_map)
    orient_pred, origin_pred, pred_cls_map, pred_reg_map = model(im)
    # vis_cls_map(gt_cls_map, pred_cls_map)

    #image = cv2.imread(im_path[0])
    #height, width, _ = image.shape

    """
    tmp = gt_bars
    print("gt_bars:", gt_bars)
    for a in range(len(tmp)):
        for idx, coord in enumerate(tmp[a]):
            tmp[a][idx] = (coord[0] * width, coord[1] * height)

    
    print("bars after: ", tmp)
    """

    bars, ticks, errorup, errordown = get_pred_bars_ticks(pred_cls_map, pred_reg_map, 0.9, 0.75)
    bars, ticks, errorup, errordown = nms(bars, ticks, errorup, errordown)

    for idx, path in enumerate(im_path):
        try:
            image = Image.open(path).convert("RGB")
        except IOError:
            raise Warning("Image file could not be opened")
        
        draw = ImageDraw.Draw(image, mode = "RGBA")

        gt_bars[idx], gt_ticks[idx], gt_errorup[idx], gt_errordown[idx] = scale([gt_bars[idx], gt_ticks[idx], gt_errorup[idx], gt_errordown[idx]], (image.width + pad[0] + pad[1], image.height + pad[2] + pad[3]))
        gt_bars[idx], gt_ticks[idx], gt_errorup[idx], gt_errordown[idx] = rmPad([gt_bars[idx], gt_ticks[idx], gt_errorup[idx], gt_errordown[idx]], pad)

        bars[idx], ticks[idx], errorup[idx], errordown[idx] = scale([bars[idx], ticks[idx], errorup[idx], errordown[idx]], (image.width + pad[0] + pad[1], image.height + pad[2] + pad[3]))
        bars[idx], ticks[idx], errorup[idx], errordown[idx] = rmPad([bars[idx], ticks[idx], errorup[idx], errordown[idx]], pad)

        drawFunction(draw, gt_orient, gt_bars[idx], gt_ticks[idx], gt_errorup[idx], gt_errordown[idx], False)
        drawFunction(draw, orient_pred, bars[idx], ticks[idx], errorup[idx], errordown[idx], True)

        image.save(os.path.join(vis_dir, f"vis_bar_char_{outlabel:06d}.png"))
        outlabel += 1

    # NOTE: In dev until the scaling conversion issue is fixed
    """
    for path in im_path:    
        readeval(bars, ticks, errorup, errordown, path)
    """
        
    for i, path in enumerate(im_path):
        barP, barR, tickP, tickR, euP, euR, edP, edR = evaluate_pts(gt_bars[i], gt_ticks[i], gt_errorup[i], gt_errordown[i], bars[i], ticks[i], errorup[i], errordown[i], eval_thresh)
        barF1, tickF1, euF1, edF1 = f1(barP, barR), f1(tickP, tickR), f1(euP, euR), f1(edP, edR)
        im_metrics.append((barP, barR, barF1, tickP, tickR, tickF1, euP, euR, euF1, edP, edR, edF1))    

mbP = sum([im[0] for im in im_metrics]) / len(im_metrics)
mbR = sum([im[1] for im in im_metrics]) / len(im_metrics)
mbF1 = sum([im[2] for im in im_metrics]) / len(im_metrics)
mtP = sum([im[3] for im in im_metrics]) / len(im_metrics)
mtR = sum([im[4] for im in im_metrics]) / len(im_metrics)
mtF1 = sum([im[5] for im in im_metrics]) / len(im_metrics)
meuP = sum([im[6] for im in im_metrics]) / len(im_metrics)
meuR = sum([im[7] for im in im_metrics]) / len(im_metrics)
meuF1 = sum([im[8] for im in im_metrics]) / len(im_metrics)
medP = sum([im[9] for im in im_metrics]) / len(im_metrics)
medR = sum([im[10] for im in im_metrics]) / len(im_metrics)
medF1 = sum([im[11] for im in im_metrics]) / len(im_metrics)
avgF1 = (mbF1 + mtF1 + meuF1 + medF1) / 4.

print(f'{0.9:.2f}, {0.75:.2f}, ' +
        f'{eval_thresh:.1f}, '+
        f'{mbP:.4f}, {mbR:.4f}, {mtP:.4f}, {mtR:.4f}, {meuP:.4f}, {meuR:.4f}, {medP:.4f}, {medR:.4f}, '+
        f'{mbF1:.4f}, {mtF1:.4f}, {meuF1:.4f}, {medF1:.4f}, {avgF1:.6f}')
    

