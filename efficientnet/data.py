from __future__ import print_function, division
import os
import cv2
import json
import csv
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from PIL import ImageFilter

import code

from hparams import hparams

# Performs the horizontal flip of the image.
def hflip_img(image):
    # Load raw image file into memor
    res = cv2.flip(image, 1) # Flip the image
    return res


# Performs a vertical flip of the image.
def vflip_img(image):
    # Load raw image file into memor
    res = cv2.flip(image, 0) # Flip the image
    return res


# Rotates the image given a specific number of degrees, positive is clockwise
# negative is counterclockwise.
def rotate_img(image, angle):
    rows,cols,_ = image.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    res = cv2.warpAffine(image,M,(cols,rows))
    return res


# Translates the image horizontally and vertically, postivie is down and right
# negative is up and left.
def shift_img(image, x, y):
    rows,cols,_ = image.shape

    M = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst


# Blurs the image using the average value from the 5X5 pixle square surrounding
# each pixel.
def blur_img(image, size = 5):
    blur = cv2.blur(image,(size,size))
    return blur


# Blurs the image using Gaussian weights from the 5X5 pixle square surrounding
# each pixel.
def gauss_img(image, size = 5):
    blur = cv2.GaussianBlur(image,(size,size), 0)
    return blur


# Applys a bilateral filter that sharpens the edges while bluring the other areas.
def bilateral_img(image, size = 5):
    blur = cv2.bilateralFilter(image,9,75,75)
    return blur

# Crops image from borders 
def crop_img(image):
    rows, cols, _ = image.shape
    x1, x2, y1, y2 = np.random.randint(20), np.random.randint(20), np.random.randint(20),  np.random.randint(20)
    crop = image[x1:rows-x2, y1:cols-y2]
    return crop



class ChestData(Dataset):

  def __init__(self, data_csv, data_dir, transform=None, image_shape=hparams.image_shape, pre_process=None, ds_type='', augment=0):
        'Initialization'
        self.data_csv = data_csv
        self.data_dir = data_dir
        self.image_shape = hparams.image_shape
        self.ds_type = ds_type
        self.augment = augment
        self.transform = transform
        self.pre_process = pre_process
        self.data_frame = pd.read_csv(data_csv)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_frame)

  def __getitem__(self, index):
        'Generates one sample of data'

        label = torch.tensor([self.data_frame.iloc[index].loc[hparams.id_to_class[idx]] for idx in  hparams.id_to_class.keys()])
        for idx in range(label.shape[0]):
            if label[idx] == -1 and hparams.lsr == 1:
                label[idx] = np.random.uniform(0.55, 0.85, size = 1)[0]
            if label[idx] == -1 and hparams.lsr == 0:
                label[idx] = np.random.uniform(0, 0.30, size = 1)[0]
        img_name = os.path.join(self.data_dir,
                                self.data_frame.iloc[index, 0])
        
        image = cv2.imread(img_name)
        if self.augment > 0:
            if np.random.uniform(0, 1.0) > 0.7:
                image = hflip_img(image)
#             if np.random.uniform(0, 1.0) > 0.5:
#                 image = rotate_img(image, np.random.randint(15))
#             if np.random.uniform(0, 1.0) > 0.7:
#                 image = shift_img(image, np.random.randint(10), np.random.randint(10))
        if self.augment > 1:
            if np.random.uniform(0, 1.0) > 0.5:
                image = crop_img(image)
            if np.random.uniform(0, 1.0) > 0.7:
                if np.random.uniform(0, 1.0) < 0.33:
                    image = blur_img(image)
                elif np.random.uniform(0, 1.0) < 0.66:
                    image = gauss_img(image)
                else:
                    image = bilateral_img(image)
        image = Image.fromarray(image).convert('RGB')
        image = image.resize(hparams.image_shape, Image.ANTIALIAS)

        if self.transform:
            image = self.transform(image)

        return (image, label, self.data_frame.iloc[index, 0])
