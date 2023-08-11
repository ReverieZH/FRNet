"""
 @Time    : 2022/6/7 20:11
 @Author  : Zehua Ren
 @E-mail  :
 @Project : FRNet
 @File    : train.py
 @Function: Training
"""
import datetime
import time
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import joint_transforms_freq
from config import cod_training_root, cod10k_path
from config import efficientnet_backbone_path
from datasets_freq import ImageFolder
from misc import AvgMeter, check_mkdir, calculate_sad_mse_mad_whole_img
from FRNet import FRNet
import logging
import loss
import torch_dct as DCT
# from apex import amap
cudnn.benchmark = True
torch.manual_seed(2022)
device_ids = [0]
ckpt_path = './ckpt'
exp_name = 'FRNet'

args = {
    'epoch_num': 120,
    'train_batch_size': 16,
    'last_epoch': 0,
    'lr': 1e-4,
    'clip': 0.5,
    'decay_rate': 0.8,
    'decay_epoch': 50,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': False,
    'scale': 416,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'Adam',
    'val_epoch': 1,
    'only_val': False
}

print(torch.__version__)

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
model_dir = os.path.join(ckpt_path, exp_name)
print(model_dir)
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)

log_path = os.path.join(ckpt_path, exp_name, 'log', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.logs')
if os.path.exists(log_path) is False:
    file = open(log_path, 'w', encoding='utf-8')
    file.close()

# Transform Data.
joint_transform = joint_transforms_freq.Compose([
    joint_transforms_freq.RandomHorizontallyFlip(),
    joint_transforms_freq.Resize((args['scale'], args['scale']))
])

edge_joint_transform = joint_transforms_freq.Edge_Compose([
    joint_transforms_freq.Edge_RandomHorizontallyFlip(),
    joint_transforms_freq.Edge_Resize((args['scale'], args['scale']))
])

img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_join_transform = joint_transforms_freq.Compose([
    joint_transforms_freq.Resize((args['scale'], args['scale']))
])
test_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(cod_training_root, edge_joint_transform, img_transform, target_transform, edge=True, freq=True)
test_set = ImageFolder(cod10k_path, test_join_transform, test_img_transform, target_transform, freq=True)

print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=6, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, num_workers=6, shuffle=False, pin_memory=True)

total_epoch = args['epoch_num'] * len(train_loader)
# loss function
adaptive_loss = loss.adaptive_loss().cuda(device_ids[0])
att_loss = loss.AttentionLossSingleMap().cuda(device_ids[0])
edge_loss = loss.EdgeLoss().cuda(device_ids[0])


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    if 20 <= epoch < 40:
        lr = init_lr * 0.5
    elif epoch <= 40:
        lr = init_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def main():
    print(args)
    print(exp_name)
    # logging.basicConfig(filename=log_path, level=logging.INFO)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='UTF-8')
    logger.addHandler(file_handler)
    args['log'] = logging
    logger.info("===============================")
    logger.info(f"===> Loading args\n{args}")
    logger.info("===> Environment init")
    logger.info('===> Building the model')
    net = FRNet(efficientnet_backbone_path, backbone="EfficientNet").cuda(device_ids[0]).train()
    # net = FRNet(res2net_backbone_path, backbone="res2net").cuda(device_ids[0]).train()
    logger.info('===> Initialize optimizer')
    if args['optimizer'] == 'Adam':
        print("Adam")
        logger.info('===> optimizer Adam')
        # optimizer = torch.optim.Adam(net.parameters(), args['lr'], weight_decay=5e-4, betas=(0.5, 0.999))
        optimizer = torch.optim.Adam(net.parameters(), args['lr'])
    else:
        print("SGD")
        logger.info('===> optimizer SGD')
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])
    bestmad = 10
    if args['snapshot']:
        print('Training Resumes From snapshot ')
        tmp = torch.load(os.path.join(ckpt_path, exp_name, 'best.pth'))
        state = tmp['state']
        epoch = tmp['epoch']
        mad = tmp['mad']
        bestmad = mad
        print("snapshot mad ", mad)
        args['last_epoch'] = epoch
        net.load_state_dict(state)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    if args['only_val']:
        print('only_val')
        print('load model:', os.path.join(ckpt_path, exp_name, 'best.pth'))
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'best.pth')))
        val(net, logger)
        return None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args['decay_rate'], patience=3, min_lr=1e-5)
    train(net, optimizer, scheduler, logger, best_mad=bestmad)

def train(net, optimizer, scheduler, logger, best_mad=None):
    curr_iter = 1
    start_time = time.time()
    num_iter = len(train_loader)
    if best_mad is None:
        best_mad = 10
    best_epoch = 0
    early_stopping = 0
    for epoch in range(args['last_epoch'] + 1, args['epoch_num']):
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record, loss_all_record, MAD_record, all_MAD_record, loss_edge_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_each_epoch = []
        sad_epoch, mse_epoch, mad_epoch, all_mad_epoch, all_mse_epoch = 0, 0, 0, 0, 0
        train_iterator = tqdm(train_loader, total=len(train_loader))
        net.train()
        sample_num = 0
        for iteration, data in enumerate(train_iterator, 1):
            inputs, ycbcr_image, labels, edge, img_path = data
            batch_size = inputs.size(0)
            size = inputs.size(2)
            sample_num += inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])
            edge = Variable(edge).cuda(device_ids[0])
            ycbcr_image = Variable(ycbcr_image).cuda(device_ids[0])
            ycbcr_image = ycbcr_image.reshape(batch_size, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5).contiguous()  # b h/8 w/8 3 8 8
            ycbcr_image = DCT.dct_2d(ycbcr_image, norm='ortho')
            ycbcr_image = ycbcr_image.reshape(batch_size, size // 8, size // 8, -1).permute(0, 3, 1, 2).contiguous()  # b 3*8*8 h/8 w/8
            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4, pred_all, freq_output_3, pred_edge = net(inputs, ycbcr_image)

            loss_1 = adaptive_loss(predict_1, labels)
            loss_2 = adaptive_loss(predict_2, labels) + adaptive_loss(freq_output_3, labels)
            loss_3 = adaptive_loss(predict_3, labels)
            loss_4 = adaptive_loss(predict_4, labels)

            loss_all = adaptive_loss(pred_all, labels)
            loss_edge = att_loss(pred_edge, edge)


            loss = 1 * loss_1 + 2 * loss_2 + 4 * loss_3 + 8 * loss_4 + 8 * loss_edge + 8 * loss_all


            loss.backward()
            clip_gradient(optimizer, args['clip'])
            optimizer.step()
            sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(pred_all, labels)
            loss_record.update(loss.cpu().data.numpy(), 1)
            loss_1_record.update(loss_1.cpu().data.numpy(), 1)
            loss_2_record.update(loss_2.cpu().data.numpy(), 1)
            loss_3_record.update(loss_3.cpu().data.numpy(), 1)
            loss_4_record.update(loss_4.cpu().data.numpy(), 1)
            loss_edge_record.update(loss_edge.cpu().data.numpy(), 1)
            MAD_record.update(mad_diff, sample_num)
            loss_each_epoch.append(loss.cpu().item())
            sad_epoch += sad_diff
            mse_epoch += mse_diff
            mad_epoch += mad_diff
            logger.info(
                "PFNet-Epoch[{}/{}]({}/{})  Lr:{:.8f} Loss:{:.5f}  mad_diff:{:5f} ".format(
                    epoch, args['last_epoch'] + 1 + args['epoch_num'], iteration, num_iter, optimizer.param_groups[0]['lr'], loss.data,
                     mad_epoch / sample_num))
            log = 'epoch: [%3d], [%6d]/[%6d], lr[%.6f], loss[%.5f], loss_1[%.5f], loss_2[%.5f], loss_3[%.5f], loss_4[%.5f], MAD[%.5f]' % \
                  (epoch, iteration, num_iter, optimizer.param_groups[0]['lr'], loss_record.avg, loss_1_record.avg,
                   loss_2_record.avg, loss_3_record.avg, loss_4_record.avg, mad_epoch / sample_num)
            train_iterator.set_description(log)
            curr_iter += 1
        if epoch % args['val_epoch'] == 0:
            mad = val(net, logger, epoch)
            if mad < best_mad:
                net.cpu()
                torch.save({'epoch': epoch, 'state': net.state_dict(), "mad": mad}, os.path.join(model_dir, 'best.pth'))
                print('save the best result on epoch', epoch)
                best_mad = mad
                best_epoch = epoch
                early_stopping = 0
                net.cuda(device_ids[0])
            else:
                early_stopping += 1

            if early_stopping == 10:
                print("early stopping in epoch:", epoch)
                print('save the best result:', best_mad, ' on epoch:', best_epoch)
                break
            scheduler.step(mad)
        if epoch in args['save_point']:
            net.cpu()
            torch.save({'epoch': epoch, 'state': net.state_dict(), "mad": mad}, os.path.join(model_dir, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save({'epoch': epoch, 'state': net.state_dict(), "mad": mad}, os.path.join(model_dir, '%d.pth' % epoch))
            net.cuda(device_ids[0])
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            print(" the best epoch is ", best_epoch, "the best mad is ", best_mad)
            return
    print('save the best result:', best_mad, ' on epoch:', best_epoch)

def val(net, logger, epoch):
    net.eval()
    total_number = len(test_loader)
    logger.info("===============================")
    logger.info(f'====> Start Testing \t--Dataset: COD10k_Test \t --Number: {total_number}')
    mse_diffs = 0.
    mad_diffs = 0.
    test_iterator = tqdm(test_loader)
    sample_num = 0
    for iteration, data in enumerate(test_iterator, 1):
        inputs, ycbcr_image, labels, img_path = data
        batch_size = inputs.size(0)
        size = inputs.size(2)
        sample_num += inputs.size(0)
        inputs = Variable(inputs).cuda(device_ids[0])
        labels = Variable(labels).cuda(device_ids[0])
        ycbcr_image = Variable(ycbcr_image).cuda(device_ids[0])
        ycbcr_image = ycbcr_image.reshape(batch_size, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)

        ycbcr_image = DCT.dct_2d(ycbcr_image, norm='ortho')
        ycbcr_image = ycbcr_image.reshape(batch_size, size // 8, size // 8, -1).permute(0, 3, 1, 2)


        predict4, predict3, predict2, predict1, pred_all,  freq_output_3, pred_edge = net(inputs, ycbcr_image)
        sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(pred_all, labels, mode='test')
        mse_diffs += mse_diff
        mad_diffs += mad_diff

        logger.info(
            f"Testing numbers:{iteration}/{total_number}\t image:{img_path}\tMSE: {mse_diff} \t MAD: {mad_diff} \t")
        log = 'Testing  [%3d], MSE[%.5f], MAD[%.5f]' % \
              (iteration, mse_diffs / sample_num, mad_diffs / sample_num)
        test_iterator.set_description(log)
    logger.info("===============================")
    logger.info(
        f"The Testing Result: \t MSE: {mse_diffs / sample_num} \t MAD: {mad_diffs / sample_num} \t ")
    print(f"The Testing Result: \t MSE: {mse_diffs / sample_num} \t MAD: {mad_diffs / sample_num} \t")
    return float(mad_diffs / sample_num)


if __name__ == '__main__':
    main()