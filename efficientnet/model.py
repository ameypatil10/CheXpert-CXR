from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision
from efficientnet_pytorch import EfficientNet

from hparams import hparams

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b1', advprop=True, num_classes=hparams.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
