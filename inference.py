import os
import sys
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import scipy.special
import tqdm
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

from lanedet.model.model import parsingNet
from lanedet.utils.common import merge_config
from lanedet.utils.dist_utils import dist_print
from lanedet.data.constant import culane_row_anchor, tusimple_row_anchor

def init_model(cfg, device='cuda:0'):
    assert cfg.backbone in ['18','34','50','101','152',
        '50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        cfg.row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        cfg.row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
        use_aux=False).to(device)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    net.cfg = cfg
    net.device = device

    return net

def inference_model(model, img):
    cfg = model.cfg
    im_w = 800
    im_h = 288
    img_transforms = transforms.Compose([
        transforms.Resize((im_h, im_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    else:
        raise ValueError('img should be str, ndarray or Image')
    ori_im_w, ori_im_h = img.size

    img = img_transforms(img)
    img = img[None]
    img = img.to(model.device)

    with torch.no_grad():
        out = model(img)

    col_sample = np.linspace(0, im_w - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].data.cpu().numpy()
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc

    out = np.zeros((model.num_lanes, len(cfg.row_anchor), 2), dtype=int)
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    out[i, k] = (int(out_j[k, i]
                                     * col_sample_w
                                     * ori_im_w
                                     / im_w) - 1,
                                 int(ori_im_h
                                     * cfg.row_anchor[k]
                                     / im_h) - 1)
    return out

def show_result(img, result, **kwargs):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise ValueError('img should be str, ndarray or Image')
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    draw = ImageDraw.Draw(img)
    for cls, points in enumerate(result):
        valid_points = points[points[:, 0] > 0]
        if len(valid_points) > kwargs.get('p_thre', 2):
            for point in valid_points:
                draw.ellipse([tuple(point), tuple(point+10)], fill=colors[cls],
                    width=1)
    return img


if __name__ == "__main__":
    args, cfg = merge_config()
    dist_print('start testing...')
    net = init_model(cfg, 'cuda:0')
    result = inference_model(net, args.test_img)
    img = show_result(args.test_img, result)
    img.save(os.path.join(args.test_work_dir, os.path.basename(args.test_img)))
