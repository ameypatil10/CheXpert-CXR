import time
import code
import os, torch, sys
import torch
import csv
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from skimage.util import random_noise
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from hparams import hparams
from data import ChestData
from model import Discriminator
from metric import accuracy_metrics

epsilon = 0.0000000001

plt.switch_backend('agg')

def test(model_paths, data=(hparams.valid_csv, hparams.valid_dir), plot_auc='valid', plot_path=hparams.result_dir+'valid', best_thresh=None, pred_csv=None):

    test_dataset = ChestData(data_csv=data[0], data_dir=data[1], augment=hparams.TTA,
                        transform=transforms.Compose([
                            transforms.Resize(hparams.image_shape),
                            transforms.ToTensor(),
#                             transforms.Normalize((0.5027, 0.5027, 0.5027), (0.2915, 0.2915, 0.2915))
                        ]))

    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size,
                            shuffle=False, num_workers=4)


    discriminators = [Discriminator().to(hparams.gpu_device) for _ in model_paths]
    if hparams.cuda:
        discriminators = [nn.DataParallel(discriminators[i], device_ids=hparams.device_ids) for i in range(len(model_paths))]
    checkpoints = [torch.load(model_path, map_location=hparams.gpu_device) for model_path in model_paths]
    for i in range(len(model_paths)):
        discriminators[i].load_state_dict(checkpoints[i]['discriminator_state_dict'])

    def put_eval(model):
        model = model.eval()
        model.training = hparams.eval_dp_on
        return model
    discriminators = [put_eval(discriminator) for discriminator in discriminators]
    # print('Model loaded')

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    print('Testing model on {0} examples. '.format(len(test_dataset)))

    with torch.no_grad():
        pred_logits = torch.zeros((len(test_dataset), hparams.num_classes))
        if hparams.cuda:
            pred_logits = pred_logits.to(hparams.gpu_device)
        for _ in range(hparams.repeat_infer):
            labels_list = []
            img_names_list = []
            pred_logits_list = []
            for (img, labels, img_names) in tqdm(test_loader):
                img = Variable(img.float(), requires_grad=False)
                labels = Variable(labels.float(), requires_grad=False)
                if hparams.cuda:
                    img_ = img.to(hparams.gpu_device)
                    labels = labels.to(hparams.gpu_device)
                pred_logits_ = discriminators[0](img_)
                pred_logits_ = pred_logits_*0
                for discriminator in discriminators:
                    pred_logits_ += discriminator(img_)
                pred_logits_ = 1.0*pred_logits_/len(model_paths)

                pred_logits_list.append(pred_logits_)
                labels_list.append(labels)
                img_names_list += list(img_names)

            pred_logits += torch.cat(pred_logits_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
        
        pred_logits = 1.0*pred_logits/hparams.repeat_infer
        _, pred_labels = torch.max(pred_logits, axis=1)

        f1, acc, conf_mat = accuracy_metrics(labels, pred_labels)

        print('== Test on -- '+str(model_paths)+' == \n f1 - {0:.4f}, acc - {1:.4f}'\
            .format(f1, acc))
    return f1
