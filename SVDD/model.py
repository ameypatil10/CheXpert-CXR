from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict
import torchvision

from hparams import hparams


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up = nn.Sequential(OrderedDict([
            # 4X4
            ('derelu0', nn.ReLU()),
            ('interp0', nn.UpsamplingNearest2d(scale_factor=2)),
            ('deconv0', nn.ConvTranspose2d(in_channels=128, out_channels=64, bias=False, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1,1))),
            # 8X8
            ('debn0', nn.BatchNorm2d(num_features=64, affine=False)),

            ('derelu1', nn.ReLU()),
            ('interp1', nn.UpsamplingNearest2d(scale_factor=2)),
            # 16X16
            ('deconv1', nn.ConvTranspose2d(in_channels=64, out_channels=32, bias=False, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1,1))),
            # 32X32
            ('debn1', nn.BatchNorm2d(num_features=32, affine=False)),

            ('derelu2', nn.ReLU()),
            ('interp2', nn.UpsamplingNearest2d(scale_factor=2)),
            # 64X64
            ('deconv2', nn.ConvTranspose2d(in_channels=32, out_channels=16, bias=False, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1,1))),
            # 128X128
            ('debn2', nn.BatchNorm2d(num_features=16, affine=False)),

            ('derelu3', nn.ReLU()),
            # ('interp3', nn.UpsamplingNearest2d(scale_factor=2)),
            # 512X512
            ('deconv3', nn.ConvTranspose2d(in_channels=16, out_channels=hparams.num_channel, bias=False, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1,1))),
            # 1024X1024/2
        ]))


    def forward(self, x):
        x = x.view(x.shape[0], 128, 4, 4)
        x = torch.sigmoid(self.up(x))
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down = nn.Sequential(OrderedDict([
            # 1024X1024/2
            ('conv1', nn.Conv2d(in_channels=hparams.num_channel, out_channels=16, bias=False, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
            # 512X512/2
            ('bn1', nn.BatchNorm2d(num_features=16, eps=1e-04, affine=False)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2, 2)),
            # 256X256/2

            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, bias=False, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
            # 128X128/2
            ('bn2', nn.BatchNorm2d(num_features=32, eps=1e-04, affine=False)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2, 2)),
            # 64X64/2

            ('conv3', nn.Conv2d(in_channels=32, out_channels=64, bias=False, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
            # 32X32/2
            ('bn3', nn.BatchNorm2d(num_features=64, eps=1e-04, affine=False)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool2d(2, 2)),
            # 16X16/2

            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, bias=False, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
            # 8X8/2
            ('bn4', nn.BatchNorm2d(num_features=128, eps=1e-04, affine=False)),
            ('relu4', nn.ReLU()),
            # ('pool4', nn.MaxPool2d(2, 2)),
            # 4X4
        ]))
        self.fc = nn.Linear(128*4*4, hparams.latent_dim)

        self.center = None
        self.radius = 0


    def forward(self, x):
        x = self.down(x)
        x = x.view(x.shape[0],-1)
        z = self.fc(x)
        return z

#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.up = nn.Sequential(OrderedDict([
#             # 3X3
#             ('derelu0', nn.ReLU()),
#             ('interp0', nn.UpsamplingNearest2d(size=(7,7))),
#             # 7X7
#             ('deconv0', nn.ConvTranspose2d(in_channels=128, out_channels=64, bias=False, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1,1))),
#             # 14X14
#             ('debn0', nn.BatchNorm2d(num_features=64, affine=False)),
#
#             ('derelu1', nn.ReLU()),
#             ('interp1', nn.UpsamplingNearest2d(scale_factor=2)),
#             # 28X28
#             ('deconv1', nn.ConvTranspose2d(in_channels=64, out_channels=32, bias=False, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1,1))),
#             # 56X56
#             ('debn1', nn.BatchNorm2d(num_features=32, affine=False)),
#
#             ('derelu2', nn.ReLU()),
#             ('interp2', nn.UpsamplingNearest2d(scale_factor=2)),
#             # 112X112
#             ('deconv2', nn.ConvTranspose2d(in_channels=32, out_channels=hparams.num_channel, bias=False, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1,1))),
#             # 224X224
#         ]))
#
#
#     def forward(self, x):
#         x = x.view(x.shape[0], 128, 3, 3)
#         x = torch.sigmoid(self.up(x))
#         return x
#
#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.down = nn.Sequential(OrderedDict([
#             # 224X224
#             ('conv1', nn.Conv2d(in_channels=hparams.num_channel, bias=False, out_channels=32, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
#             # 112X112
#             ('bn1', nn.BatchNorm2d(num_features=32, eps=1e-04, affine=False)),
#             ('relu1', nn.ReLU()),
#             ('pool1', nn.MaxPool2d(2, 2)),
#             # 56X56
#
#             ('conv2', nn.Conv2d(in_channels=32, out_channels=64, bias=False, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
#             # 28X28
#             ('bn2', nn.BatchNorm2d(num_features=64, eps=1e-04, affine=False)),
#             ('relu2', nn.ReLU()),
#             ('pool2', nn.MaxPool2d(2, 2)),
#             # 14X14
#
#             ('conv3', nn.Conv2d(in_channels=64, out_channels=128, bias=False, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
#             # 7X7
#             ('bn3', nn.BatchNorm2d(num_features=128, eps=1e-04, affine=False)),
#             ('relu3', nn.ReLU()),
#             ('pool3', nn.MaxPool2d(2, 2)),
#             # 3X3
#
#             # ('conv4', nn.Conv2d(in_channels=256, out_channels=512, bias=False, kernel_size=5, stride=2, padding=(2, 2), padding_mode='reflect')),
#             # # 14X14
#             # ('bn4', nn.BatchNorm2d(num_features=512, eps=1e-04, affine=False)),
#             # ('relu4', nn.ReLU()),
#         ]))
#         self.fc = nn.Linear(128*3*3, hparams.latent_dim)
#
#         self.center = None
#         self.radius = 0
#
#
#     def forward(self, x):
#         x = self.down(x)
#         x = x.view(x.shape[0],-1)
#         z = self.fc(x)
#         return z
