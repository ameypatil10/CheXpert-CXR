from __future__ import print_function, division
import os
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

def create_csv():
    content = list(map(lambda x: 'normal/'+x, os.listdir('../data/dataset_allinfected/train/normal/')))
    data = {'images': content, 'labels': [1]*len(content)}
    df = pd.DataFrame(data)
    df.to_csv('../data/dataset_allinfected/train/train.csv', index=False)

    content = list(map(lambda x: 'normal/'+x, os.listdir('../data/dataset_allinfected/val/normal/')))
    labels = [1]*len(content)
    content1 = list(map(lambda x: 'infected/'+x, os.listdir('../data/dataset_allinfected/val/infected/')))
    labels += [0]*len(content1)
    data = {'images': content+content1, 'labels': labels}
    df = pd.DataFrame(data)
    df.to_csv('../data/dataset_allinfected/val/valid.csv', index=False)

    content = list(map(lambda x: 'normal/'+x, os.listdir('../data/dataset_allinfected/test/normal/')))
    labels = [1]*len(content)
    content1 = list(map(lambda x: 'infected/'+x, os.listdir('../data/dataset_allinfected/test/infected/')))
    labels += [0]*len(content1)
    data = {'images': content+content1, 'labels': labels}
    df = pd.DataFrame(data)
    df.to_csv('../data/dataset_allinfected/test/test.csv', index=False)

# create_csv()

def one_class_csv():
    df = pd.read_csv('../data/NIH_curated/train_curated.csv')
    df = df.loc[df['Infected'] == False]
    df = df[['Image Index', 'Infected']]
    df.to_csv(hparams.train_csv, index=False)
    df = pd.read_csv('../data/NIH_curated/val_curated.csv')
    df = df[['Image Index', 'Infected']]
    df.to_csv(hparams.valid_csv, index=False)
    df = pd.read_csv('../data/NIH_curated/test_curated.csv')
    df = df[['Image Index', 'Infected']]
    df.to_csv(hparams.test_csv, index=False)
    print('One class data ready.')

# one_class_csv()

class ChestData(Dataset):

  def __init__(self, data_csv, data_dir, transform=None, image_shape=hparams.image_shape, pre_process=None, ds_type=''):
        'Initialization'
        self.data_csv = data_csv
        self.data_dir = data_dir
        self.image_shape = hparams.image_shape
        self.ds_type = ds_type
        self.transform = transform
        self.pre_process = pre_process
        self.data_frame = pd.read_csv(data_csv)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_frame)

  def __getitem__(self, index):
        'Generates one sample of data'

        label = self.data_frame.iloc[index, 2]
        img_name = os.path.join(self.data_dir,
                                self.data_frame.iloc[index, 0])

        if self.pre_process:
            image = process_image(img_name)
        else:
            image = Image.open(img_name)
            # image = image.convert("RGB")
        image = image.resize(hparams.image_shape, Image.ANTIALIAS)

        if self.transform:
            image = self.transform(image)
            image = image[0,:,:].reshape(hparams.num_channel, hparams.image_shape[0], hparams.image_shape[1])
        # print(img_name, image.shape)

        return (image, label, img_name)
