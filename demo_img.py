import os
from PIL import Image

import scipy.special
import tqdm
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from data.constant import culane_row_anchor, tusimple_row_anchor

def init_model(cfg, args):
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
        use_aux=False).cuda(args.local_rank)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    return net


if __name__ == "__main__":
    args, cfg = merge_config()
    dist_print('start testing...')
    net = init_model(cfg, args)

    im_w = 800
    im_h = 288
    img_transforms = transforms.Compose([
        transforms.Resize((im_h, im_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    output_dir = args.test_work_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_path = args.test_img

    img = Image.open(img_path)
    img = img_transforms(img)
    img = img[None]
    img = img.cuda(args.local_rank)

    with torch.no_grad():
        out = net(img)

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

    vis = cv2.imread(img_path)
    ori_im_h, ori_im_w = vis.shape[:2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    pos = (int(out_j[k, i] 
                                * col_sample_w 
                                * ori_im_w 
                                / im_w) - 1,
                           int(ori_im_h * cfg.row_anchor[k] / im_h) - 1)
                    cv2.circle(vis, pos, 5, colors[i], -1)

    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), vis)
