import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.efficientnet.EfficientNet import EfficientNet
from backbone.efficientnet.effi_utils import get_model_shape

from utils import *

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

    def initialize(self):
        weight_init(self)

class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat,eps=1e-5,))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    def initialize(self):
        weight_init(self)

class ForegroundRefinement(nn.Module):
    def __init__(self, channel1, channel2, edge=False):
        """
        256, 512
        :param channel1: current-level features channel
        :param channel2: higher-level features channel
        """
        super(ForegroundRefinement, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.output_map = nn.Conv2d(self.channel1, 1, 3, 1, 1)


        self.fp = RCAB(self.channel1, bn=True)
        self.fn = RCAB(self.channel1, bn=True)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.conv2d1 = nn.Conv2d(self.channel1, self.channel1, 3, stride=1, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel1, eps=1e-5)
        self.relu1 = nn.ReLU(inplace=True)

        self.fp_bn = nn.BatchNorm2d(self.channel1, eps=1e-5)
        self.fp_relu = nn.ReLU(inplace=True)

        self.bp_bn = nn.BatchNorm2d(self.channel1, eps=1e-5)
        self.bp_relu = nn.ReLU(inplace=True)

        self.conv2d2 = nn.Conv2d(self.channel1, self.channel1, 3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channel1, eps=1e-5)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv = BasicConv2d(self.channel2, self.channel1, 3, 1, 1)
        if edge:
            self.edge = edge
            self.edge_weight = nn.Parameter(torch.ones(1))


    def forward(self, x, y, in_map, need_up=True, edge_att=None, denoise=0.93):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        if need_up:
            # up = self.up(y)
            if self.channel1 != self.channel2:
                y = self.conv(y)
            up = nn.functional.interpolate(y, mode='bilinear', align_corners=True, scale_factor=2)
            input_map = torch.sigmoid(nn.functional.interpolate(in_map, mode='bilinear', align_corners=True, scale_factor=2))
            # input_map = self.input_map(in_map)
        else:
            up = y
            input_map = torch.sigmoid(in_map)
        high_f_feature = up * input_map
        high_b_feature = up * (1 - input_map)

        f_feature = x + high_f_feature
        b_feature = x + high_b_feature

        fp = self.alpha * self.fp(f_feature)
        fn = 1 - self.beta * self.fn(b_feature)
        refine1 = fp + fn
        if self.edge:
            edge_att = F.interpolate(edge_att, size=x.size()[2:], mode='bilinear', align_corners=True)
            refine1 = self.edge_weight * refine1 * edge_att + refine1
            refine1 = self.conv2d1(refine1)
            refine1 = self.bn1(refine1)
            refine1 = self.relu1(refine1)
        else:
            refine1 = self.conv2d1(refine1)
            refine1 = self.bn1(refine1)
            refine1 = self.relu1(refine1)
        refine2 = self.conv2d2(refine1)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        output_map = self.output_map(refine2)
        return refine2, output_map

    def initialize(self):
        weight_init(self)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)

class SpatialGate(nn.Module):
    """generation spatial attention mask"""

    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x).abs()
        # return torch.sigmoid(x)
        return x
    def initialize(self):
        weight_init(self)


class ChannelGate(nn.Module):
    """generation channel attention mask"""

    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x).abs()
        return x
    def initialize(self):
        weight_init(self)


class BAM(nn.Module):
    def __init__(self):
        super(BAM, self).__init__()
        self.inplanes = 256
        self.res1_up = nn.Sequential(
            nn.Conv2d(48, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        # res2_up
        self.res2_up = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # res3_up
        self.res3_up = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # res4_up
        self.res4_up = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # res5_up
        self.res5_up = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.conv14 = BasicConv2d(64 + 64, 64, 3, 1, 1)
        self.conv014 = BasicConv2d(64 + 48, 64, 3, 1, 1)
        self.conv_out = self.Conv_Stage(64 + 10, [32], bias=False, output_map=True)
    def Conv_Stage(self, input_dim, dim_list, bias=True, output_map=False):
        num_layers = len(dim_list)
        dim_list = [input_dim] + dim_list

        layers = []
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(dim_list[i], dim_list[i + 1], kernel_size=3, bias=bias),
                nn.BatchNorm2d(dim_list[i + 1]),
                nn.ReLU(inplace=True)
            )
            layers.append(layer)

        if output_map:
            layer = nn.Conv2d(dim_list[-1], 1, kernel_size=1)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x, x1, x2, x3, x4, x5):
        res1_up = self.res1_up(x1)
        res2_up = self.res2_up(x2)
        res3_up = self.res3_up(x3)
        res4_up = self.res4_up(x4)
        res5_up = self.res5_up(x5)
        res1_up = F.interpolate(res1_up, size=x1.size()[2:], mode='bilinear',
                                     align_corners=True)  # (1, 2, 352, 352)
        res2_up = F.interpolate(res2_up, size=x1.size()[2:], mode='bilinear',
                                align_corners=True)  # (1, 2, 352, 352)
        res3_up = F.interpolate(res3_up, size=x1.size()[2:], mode='bilinear',
                                align_corners=True)  # (1, 2, 352, 352)
        res4_up = F.interpolate(res4_up, size=x1.size()[2:], mode='bilinear',
                                align_corners=True)  # (1, 2, 352, 352)
        res5_up = F.interpolate(res5_up, size=x1.size()[2:], mode='bilinear',
                                align_corners=True)  # (1, 2, 352, 352)
        x5_up = F.interpolate(x5, x2.size()[2:], mode='bilinear', align_corners=False)
        # x5与x2特征融合
        xf_concat1 = torch.cat([x2, x5_up], dim=1)
        xf_concat1 = self.conv14(xf_concat1)
        xf_concat1 = F.interpolate(xf_concat1, x1.size()[2:], mode='bilinear', align_corners=False)
        xf_concat1 = torch.cat((x1, xf_concat1), dim=1)
        xf_concat2 = self.conv014(xf_concat1)
        xf_concat2 = F.interpolate(xf_concat2, x1.size()[2:], mode='bilinear', align_corners=False)
        xf_concat_d = torch.cat([res1_up, res2_up, res3_up, res4_up, res5_up, xf_concat2], 1)
        out_depth = self.conv_out(xf_concat_d)
        return out_depth


class FFM(nn.Module):
    def __init__(self, in_dim, layer_norm=[192, 44, 44]):
        super(FFM, self).__init__()
        self.attention_c_y = ChannelGate(in_dim)
        self.attention_c_cb = ChannelGate(in_dim)
        self.attention_c_cr = ChannelGate(in_dim)

        self.conv_reduce = nn.Conv2d(in_channels=192, out_channels=in_dim, kernel_size=1)
        self.seg = self.Seg()

        self.attention_s = SpatialGate(in_dim * 2)
        self.attention_c_rgb = ChannelGate(in_dim)
        self.attention_c_freq = ChannelGate(in_dim)
        # self.initialize()

    def Seg(self):
        a = torch.zeros(1, 64, 1, 1)
        for i in range(0, 32):
            a[0, i + 32, 0, 0] = 1
        return a

    def initialize(self):
        weight_init(self)

    def forward(self, rgb, freq):
        b, c , h, w = freq.shape
        self.seg = self.seg.to(freq.device)

        feat_y_high = freq[:, 0:64, :, :] * self.seg   # 1 64 44 44
        feat_Cb_high = freq[:, 64:128, :, :] * self.seg
        feat_Cr_high = freq[:, 128:192, :, :] * self.seg

        feat_y = freq[:, 0:64, :, :]  + feat_y_high * self.attention_c_y(feat_y_high)  # 1 64 44 44
        feat_Cb = freq[:, 64:128, :, :]  + feat_Cb_high * self.attention_c_cb(feat_Cb_high)
        feat_Cr = freq[:, 128:192, :, :]  + feat_Cr_high * self.attention_c_cr(feat_Cr_high)

        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)

        freq = self.conv_reduce(feat_DCT)
        rbg_c_attmap = self.attention_c_rgb(rgb)
        freq_c_attmap = self.attention_c_freq(freq)
        rgb = rgb * rbg_c_attmap
        freq = freq * freq_c_attmap
        attmap = self.attention_s(torch.cat((rgb, freq), 1))
        rgb = attmap * rgb
        freq = attmap * freq
        out = rgb + freq
        return out


class Freq_Att(nn.Module):
    def __init__(self):
        super(Freq_Att, self).__init__()
        self.hig_seg, self.low_seg, self.index_high, self.index_low = self.Seg()
        # self.high_band = Transformer(dim=256, depth=1, heads=4, dim_head=64, mlp_dim=128 * 2, dropout=0.3)
        # self.low_band = Transformer(dim=256, depth=1, heads=4, dim_head=64, mlp_dim=128 * 2, dropout=0.3)
        # self.band = Transformer(dim=256, depth=1, heads=4, dim_head=64, mlp_dim=128 * 2, dropout=0.3)
        # self.spatial = Transformer(dim=192, depth=1, heads=2, dim_head=64, mlp_dim=64 * 2, dropout=0.3)
        self.high_band = ChannelGate(96)
        self.low_band = ChannelGate(96)
        self.band = ChannelGate(192)
        self.spatial = SpatialGate(192)
    def Seg(self):
        dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
                9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
                17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
                25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
                33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
                41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
                49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
                57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
        a = torch.zeros(1, 64, 1, 1)
        b = torch.ones(1, 64, 1, 1)
        index_high = []
        index_low = []
        for i in range(0, 32):
            index_high.append(dict[i + 32])
            index_low.append(dict[i])
            a[0, dict[i + 32], 0, 0] = 1
        return a.nonzero(), (b - a).nonzero(), index_high, index_low

    def forward(self, DCT_x):

        feat_y = DCT_x[:, 0:64, :, :]  # 1 64 44 44
        feat_Cb = DCT_x[:, 64:128, :, :]
        feat_Cr = DCT_x[:, 128:192, :, :]

        feat_y_high = feat_y[:, self.index_high, :, :]
        feat_Cb_high = feat_Cb[:, self.index_high, :, :]
        feat_Cr_high = feat_Cr[:, self.index_high, :, :]

        feat_y_low = feat_y[:, self.index_low, :, :]  # 1 64 44 44
        feat_Cb_low = feat_Cb[:, self.index_low, :, :]
        feat_Cr_low = feat_Cr[:, self.index_low, :, :]

        high = torch.cat([feat_y_high, feat_Cb_high, feat_Cr_high], 1) # 1 96 44 44
        low = torch.cat([feat_y_low, feat_Cb_low, feat_Cr_low], 1)

        del feat_y, feat_Cb, feat_Cr, feat_y_high, feat_Cb_high, feat_Cr_high, feat_y_low, feat_Cb_low,feat_Cr_low
        # high = torch.nn.functional.interpolate(high, size=(16, 16))
        # low = torch.nn.functional.interpolate(low, size=(16, 16))

        high_attn = self.high_band(high)
        low_attn = self.low_band(low)
        high = high * high_attn
        low = low * low_attn
        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)
        del y_h, b_h, r_h, y_l, b_l, r_l
        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)  # 1 192 44  44
        c_attn = self.band(feat_DCT)
        feat_DCT = feat_DCT * c_attn
        # feat_DCT = feat_DCT.transpose(1, 2)  # 1 16x16 192
        s_attn = self.spatial(feat_DCT)
        feat_DCT = feat_DCT * s_attn

        return feat_DCT
class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn1 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn2 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn3 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn4 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn5 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(inc_ch * 2,inc_ch,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(inc_ch * 2,inc_ch,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(inc_ch * 2,inc_ch,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(inc_ch * 2,inc_ch,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(inc_ch, eps=1e-5)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(inc_ch,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

###################################################################
# ########################## NETWORK ##############################
###################################################################
class FRNet(nn.Module):
    def __init__(self, backbone_path=None, backbone="EfficientNet"):
        super(FRNet, self).__init__()
        # backbone
        # resnet50 = resnet.resnet50(backbone_path)
        self.backbone = backbone
        self.model = EfficientNet.from_pretrained(f'efficientnet-b4', weights_path=backbone_path, advprop=True)
        self.block_idx, self.channels = get_model_shape('4')
        self.cr4 = nn.Sequential(nn.Conv2d(self.channels[3], 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cr3 = nn.Sequential(nn.Conv2d(self.channels[2], 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cr2 = nn.Sequential(nn.Conv2d(self.channels[1], 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cr1 = nn.Sequential(nn.Conv2d(self.channels[0], 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bam = BAM()
        self.freq = Freq_Att()

        self.FFM4 = FFM(in_dim=64, layer_norm=[192,22,22])
        self.FFM5 = FFM(in_dim=64, layer_norm=[192,11,11])

        self.freq_out_3 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_4 = nn.Conv2d(64, 1, 1, 1, 0)


        self.fr3 = ForegroundRefinement(64, 64, edge=True)

        self.fr2 = ForegroundRefinement(64, 64, edge=True)

        self.fr1 = ForegroundRefinement(64, 64, edge=True)

        self.refunet = RefUnet(1, 32)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True
        # self.initialize()

    def forward(self, x, DCT_x):
        # x: [batch_size, channel=3, h, w]
        B, C, H, W = x.size()

        layer0 = self.model.initial_conv(x)
        layer1, layer2, layer3, layer4 = self.model.get_blocks(layer0, H, W)
        x4_rfb = self.cr4(layer4)
        x3_rfb = self.cr3(layer3)  # channel -> 32
        x2_rfb = self.cr2(layer2)
        x1_rfb = self.cr1(layer1)
        edge = self.bam(x, layer0, x1_rfb, x2_rfb, x3_rfb, x4_rfb)
        edge_att = torch.sigmoid(edge)

        feat_DCT = self.freq(DCT_x)
        feat_DCT4 = torch.nn.functional.interpolate(feat_DCT, size=x3_rfb.size()[2:], mode='bilinear',
                                                    align_corners=True)
        feat_DCT5 = torch.nn.functional.interpolate(feat_DCT, size=x4_rfb.size()[2:], mode='bilinear',
                                                    align_corners=True)
        x4_rfb = self.FFM5(x4_rfb, feat_DCT5)
        x3_rfb = self.FFM4(x3_rfb, feat_DCT4)


        freq_output_4 = self.freq_out_4(x4_rfb)
        freq_output_3 = self.freq_out_3(x3_rfb)


        fr3, predict3 = self.fr3(x3_rfb, x4_rfb, freq_output_4, edge_att=edge_att)

        fr2, predict2 = self.fr2(x2_rfb, fr3, predict3, edge_att=edge_att)

        fr1, predict1 = self.fr1(x1_rfb, fr2, predict2, edge_att=edge_att)

        predict4 = F.interpolate(freq_output_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        freq_output_3 = F.interpolate(freq_output_3, size=x.size()[2:], mode='bilinear', align_corners=True)

        edge_predict = F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge_predict_att = F.interpolate(edge_att, size=x.size()[2:], mode='bilinear', align_corners=True)
        prediction = self.refunet(predict1)

        if self.training:
            return predict4, predict3, predict2, predict1, prediction, freq_output_3, edge_predict
        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(predict1),torch.sigmoid(prediction), \
            torch.sigmoid(freq_output_3), edge_predict_att



def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: ', n)
        if n[:5] == 'layer':
            pass
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.UpsamplingBilinear2d):
            pass
        elif isinstance(m, nn.Upsample):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()
        # if isinstance(m, nn.Conv2d):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        # elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        #     nn.init.ones_(m.weight)
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.Linear):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.Sequential):
        #     weight_init(m)
        # elif isinstance(m, nn.ReLU):
        #     pass
        # else:
        #     m.initialize()


if __name__ == '__main__':
    net_path = os.path.join('./ckpt', 'FRNet', 'model.pth')
    net = FRNet('./backbone/efficientnet/adv-efficientnet-b4-44fb3a87.pth')
    tmp = torch.load(net_path)
    state = tmp['state']
    epoch = tmp['state']
    mad = tmp['mad']
    net.load_state_dict(state)

    # newstate = []
    # for k in list(state.keys()):
    #     if k.startswith('eam'):  # 将‘conv_cls’开头的key过滤掉，这里是要去除的层的key
    #         val = state[k]
    #         name = k.replace('eam', 'bam')
    #         state[name] = val
    #         del state[k]
    #     if k.startswith('PAM'):
    #         val = state[k]
    #         name = k.replace('PAM', 'FFM')
    #         state[name] = val
    #         del state[k]
    # tmp['state'] = state
    # net.load_state_dict(state)
    # torch.save(tmp, './model.pth')


