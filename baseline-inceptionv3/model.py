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

        self.model = models.inception_v3(num_classes=1000, progress=True, pretrained=True, aux_logits=True)
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, hparams.num_classes),
                                    nn.Sigmoid())
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, hparams.num_classes),
                                    nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
