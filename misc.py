"""
 @Time    : 2021/7/6 11:21
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : misc.py
 @Function: Useful functions
 
"""
import numpy as np
import os
import torch

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_sad_mse_mad_whole_img(predict, mask, mode = 'train'):
    batch = predict.shape[0]
    if mode == 'train':
        predict = np.squeeze(torch.sigmoid(predict).cpu().data.numpy())
    else:
        predict = np.squeeze(predict.cpu().data.numpy())
    mask = np.squeeze(mask.cpu().data.numpy())
    pixel = predict.shape[-1] * predict.shape[-2]
    predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
    sad_diff = np.sum(np.abs(predict - mask)) / 1000
    mse_diff = np.sum((predict - mask) ** 2) / (pixel * batch)
    mad_diff = np.sum(np.abs(predict - mask)) / (pixel * batch)
    # if mode == 'test':
    #     print(predict.shape, mad_diff)

    return sad_diff, mse_diff, mad_diff
