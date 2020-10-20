from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision
from resnext import resnext101_64x4d
from vgg import vgg19

from hparams import hparams

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
#         self.model = resnext101_64x4d(num_classes=1000, pretrained='imagenet')
#         num_ftrs = self.model.last_linear.in_features
#         self.model.last_linear = nn.Sequential(
#                                     nn.Linear(num_ftrs, hparams.num_classes),
#                                     nn.Sigmoid())

        self.model = vgg19(num_classes=1000, pretrained='imagenet', progress=True)
#         print(self.model)
        num_ftrs = 512 * 7 * 7 #self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
                                    nn.Linear(num_ftrs, 4096),
                                    nn.ReLU(True),
                                    nn.Dropout(hparams.drop_rate),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(True),
                                    nn.Dropout(hparams.drop_rate),
                                    nn.Linear(4096, hparams.num_classes),
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        return x
