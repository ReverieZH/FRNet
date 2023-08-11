import os
import cv2
import numpy as np
from tqdm import tqdm
from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, MAE
from config import *
from collections import OrderedDict
from PIL import Image


to_test = OrderedDict([
                       ('CHAMELEON', chameleon_path),
                       ('CAMO', camo_path),
                       ('COD10K', cod10k_path),
                       ('NC4K', nc4k_path),
                       ])

results_path = './results'
save_name = 'FRNet'  # 保存路径名称
save_dir = os.path.join(results_path, save_name)
print(save_name)

for name, root in to_test.items():
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
    predict_path = os.path.join(save_dir, name)
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
    mae = MAE()
    wfm = WeightedFmeasure()
    sm = Smeasure()
    em = Emeasure()
    for idx, img_name in enumerate(tqdm(img_list)):

        gt = cv2.imread(os.path.join(mask_path, img_name + '.png'), 0)
        predict = cv2.imread(os.path.join(predict_path, img_name + '.png'), 0)
        h, w = gt.shape
        predict = cv2.resize(predict, (w, h))
        mae.step(predict, gt)
        wfm.step(predict, gt)
        sm.step(predict, gt)
        em.step(predict, gt)
    print("="*20, name, "="*20)
    print('mae: %.4f' % mae.get_results()['mae'])
    print('wfm: %.4f' % wfm.get_results()['wfm'])
    print('em: %.4f' % em.get_results()['em']['curve'].mean())
    print('sm: %.4f' % sm.get_results()['sm'])
    print("=" * 50)
