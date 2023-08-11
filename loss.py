"""
 @Time    : 2021/7/6 14:31
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : loss.py
 @Function: Loss
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import torch_dct as DCT

###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
        # num_pos = torch.sum(mask == 1).float()
        # num_neg = torch.sum(mask == 0).float()
        # alpha = num_neg / (num_pos + num_neg) * 1.0
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # weit_img = weit[0].detach().cpu().numpy()
        # mask_img = mask[0].detach().cpu().numpy()
        # plt.subplot(1, 3, 1)
        # plt.imshow(np.squeeze(weit_img))
        # plt.subplot(1, 3, 2)
        # plt.imshow(np.squeeze(mask_img))
        # plt.subplot(1, 3, 3)
        # plt.imshow(np.squeeze(mask_img))
        # plt.show()

        pred = torch.sigmoid(pred.float())
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        # wiou = 1 - (inter) / (union - inter)
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce+wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)

class adaptive_loss(torch.nn.Module):
    def __init__(self):
        super(adaptive_loss, self).__init__()

    def _adaptive_loss(self, pred, mask):
        w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
        w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
        w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        # weit = 1 + 0.7 * (w1 + w2 + w3)
        weit = 1 + 5 * (w1 + w2 + w3)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred = pred.float()
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        # wiou = 1 - (inter) / (union - inter )
        wiou = 1 - (inter + 1) / (union - inter + 1)


        mae = F.l1_loss(pred, mask, reduce=None)
        # # mae = F.smooth_l1_loss(pred, mask, reduce=None)
        wmae = (weit * mae).sum(dim=(2, 3)) / (weit - 1).sum(dim=(2, 3))
        # wmae = (weit * mae).sum(dim=(2, 3)) / (weit - 1).sum(dim=(2, 3))
        # return (1.0 * wbce + 1.0 * wiou).mean()
        return (0.7 * wbce + 0.7 * wiou + 0.7 * wmae).mean()
        # return (1 * wbce + 1 * wiou + 1 * wmae).mean()

    def forward(self, pred, mask):
        return self._adaptive_loss(pred, mask)

class adaptive_different_loss(torch.nn.Module):
    def __init__(self):
        super(adaptive_different_loss, self).__init__()

    def _adaptive_loss(self, pred, mask):
        w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
        w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
        w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        # weit = 1 + 5 * (w1 + w2 + w3) * mask
        weit = 1 + 0.5 * (w1 + w2 + w3) * mask
        wbce = F.binary_cross_entropy(pred, mask, reduction='mean')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        # wbce = (weit * wbce).sum(dim=(2, 3)) / (weit + 0.5).sum(dim=(2, 3))

        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        # wiou = 1 - (inter + 1) / (union - inter + 1)  # 0.035549


        mae = F.l1_loss(pred, mask, reduce=None)
        wmae = (weit * mae).sum(dim=(2, 3)) / (weit - 1).sum(dim=(2, 3))
        return (0.7 * wbce + 0.7 * wiou + 0.7 * wmae).mean()
        # return (1 * wbce + 1 * wiou + 1 * wmae).mean()

    def forward(self, pred, mask):
        return self._adaptive_loss(pred, mask)


class adaptive_pixel_intensity_loss(torch.nn.Module):
    def __init__(self):
        super(adaptive_pixel_intensity_loss, self).__init__()

    def _adaptive_pixel_intensity_loss(self, pred, mask):
        w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
        w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
        w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        omega = 1 + 0.5 * (w1 + w2 + w3) * mask

        bce = F.binary_cross_entropy(pred, mask, reduce=None)
        abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

        inter = ((pred * mask) * omega).sum(dim=(2, 3))
        union = ((pred + mask) * omega).sum(dim=(2, 3))
        aiou = 1 - (inter + 1) / (union - inter + 1)

        mae = F.l1_loss(pred, mask, reduce=None)
        amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1).sum(dim=(2, 3))

        return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()

    def forward(self, pred, mask):
        return self._adaptive_pixel_intensity_loss(pred, mask)

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def _dice_loss(self, predict, target):
        smooth = 1
        p = 2
        valid_mask = torch.ones_like(target)
        predict = torch.sigmoid(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
        loss = 1 - num / den
        return loss.mean()

    def forward(self, pred, mask):
        return self._dice_loss(pred, mask)

def attention_loss2(output,target, gamma=4, lamda=0.5):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0
    eps = 1e-14
    output = output.float()
    pred = torch.sigmoid(output)
    p_clip = torch.clamp(pred, min=eps, max=1.0 - eps)

    weight = target * alpha * (gamma ** ((1.0 - p_clip) ** lamda)) + \
             (1.0 - target) * (1.0 - alpha) * (gamma ** (p_clip ** lamda))
    # weight_before = weight[0].detach().cpu().numpy()

    # weit = torch.abs(F.avg_pool2d(target, kernel_size=3, stride=1, padding=1) - target)
    # weight = weight + weit
    weight = weight.detach()
    # weit_np = weit[0].detach().cpu().numpy()
    # weit_img = weight[0].detach().cpu().numpy()
    # mask_img = target[0].detach().cpu().numpy()
    # plt.subplot(1, 3, 1)
    # plt.imshow(np.squeeze(weit_img))
    # plt.subplot(1, 3, 2)
    # plt.imshow(np.squeeze(mask_img))
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.squeeze(weit_np))
    # plt.show()
    al_loss = F.binary_cross_entropy_with_logits(output, target, weight, reduction='none')
    # al_loss = torch.sum(al_loss)
    al_loss = al_loss.sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    # al_loss = al_loss.sum(dim=(2, 3))
    # al_loss = torch.sum(al_loss)
    # pred = torch.sigmoid(output)
    # inter = ((pred * target) * weight).sum(dim=(2, 3))
    # union = ((pred + target) * weight).sum(dim=(2, 3))
    # wiou = 1 - (inter) / (union - inter)

    # mae = F.l1_loss(pred, target, reduce=None)
    # amae = (weight * mae).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    # return (al_loss + wiou).mean()
    return al_loss.mean()

class AttentionLossSingleMap(nn.Module):
    def __init__(self, alpha=0.1, gamma=4, lamda=0.5):
        super(AttentionLossSingleMap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        total_loss = attention_loss2(output, label, gamma=self.gamma, lamda=self.lamda)
        # total_loss = total_loss.mean()
        # total_loss = total_loss / batch_size
        return total_loss

class FreqLoss(nn.Module):
    def __init__(self):
        super(FreqLoss, self).__init__()

    def forward(self, image, freqPred, label):
        freqPred = torch.sigmoid(freqPred.float())
        DCT_pred = DCT.dct_2d(image * freqPred)
        DCT_gt = DCT.dct_2d(image * label)
        # loss = F.l1_loss(DCT_pred, DCT_gt, reduction="mean")
        # loss = F.mse_loss(DCT_pred, DCT_gt, reduction="mean")
        loss = torch.norm(DCT_pred-DCT_gt, p=2, dim=0) ** 2
        return loss.mean()


class EdgeLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=4, lamda=0.5):
        super(EdgeLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.smooth = 1
        self.p = 2

    def forward(self, output, target):
        num_pos = torch.sum(target == 1).float()
        num_neg = torch.sum(target == 0).float()
        alpha = num_neg / (num_pos + num_neg) * 1.0
        eps = 1e-14
        output = output.float()
        pred = torch.sigmoid(output)
        p_clip = torch.clamp(pred, min=eps, max=1.0 - eps)

        weight = target * alpha * (self.gamma ** ((1.0 - p_clip) ** self.lamda)) + \
                 (1.0 - target) * (1.0 - alpha) * (self.gamma ** (p_clip ** self.lamda))
        weight = weight.detach()
        al_loss = F.binary_cross_entropy_with_logits(output, target, weight, reduction='none')
        al_loss = al_loss.sum(dim=(2, 3)) / weight.sum(dim=(2, 3))


        valid_mask = torch.ones_like(target)
        predict = torch.sigmoid(output)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth
        dic_loss = 1 - num / den

        return (0.5 * al_loss + 0.5 * dic_loss).mean()