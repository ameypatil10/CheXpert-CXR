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

ALL_LBLS = ['No Finding',
            'Cardiomegaly',
            'Edema',
            'Consolidation',
            'Atelectasis',
            'Pleural Effusion',
            'Enlarged Cardiomediastinum',
            'Lung Opacity',
            'Lung Lesion',
            'Pneumonia',
            'Pneumothorax',
            'Pleural Other',
            'Fracture',
            'Support Devices']

EVAL_LBLS = ['Cardiomegaly',
             'Edema',
             'Consolidation',
             'Atelectasis',
             'Pleural Effusion',]

def aggr_preds(IDs, X, mode=''):
    df_data = {'Study': list(map(lambda x: 'valid'+'/'.join(x.split('valid')[1].split('/')[:-1]), IDs))}

    for idx in range(len(ALL_LBLS)):
        df_data[ALL_LBLS[idx]] = X[:,idx].cpu().numpy()
    df = pd.DataFrame(df_data)
    if mode == 'mean':
        df = df.groupby('Study').mean().reset_index()
    elif mode == 'max':
        df = df.groupby('Study').max().reset_index()
    elif mode == 'min':
        df = df.groupby('Study').min().reset_index()
    
    return torch.tensor(np.array(df.iloc[:,1:]))


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
#         print(checkpoints[i])
        discriminators[i].load_state_dict(checkpoints[i]['discriminator_state_dict'])

    def put_eval(model):
        model = model.eval()
#         model.training = hparams.eval_dp_on
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
        print(plot_path)
             
#         pred_logits = aggr_preds(img_names_list, pred_logits, 'min')
#         labels = aggr_preds(img_names_list, labels, 'min')

        auc, f1, acc, conf_mat, best_thresh = accuracy_metrics(labels, pred_logits, plot_auc=plot_auc, plot_path=plot_path, best_thresh=best_thresh)
        if hparams.cuda:
            pred_logits = pred_logits.cpu()
        pred_logits = pred_logits.numpy()
        print(best_thresh)

        pred_labels = 1*(pred_logits > np.array(best_thresh))
        if pred_csv:
            data = {'Path': img_names_list}
            for lbl in range(14):
                data[hparams.id_to_class[lbl]] = pred_labels[:,lbl]
            df = pd.DataFrame(data)
            df.to_csv('../results/predictions_{}.csv'.format(pred_csv), index=False)
            print('predictions saved to "../results/predictions_{}.csv"'.format(pred_csv))

        print('== Test on -- '+str(model_paths)+' == \n\
            auc_{0} - {5:.4f}, auc_{1} - {6:.4f}, auc_{2} - {7:.4f}, auc_{3} - {8:.4f}, auc_{4} - {9:.4f}, auc_micro - {10:.4f}, auc_macro - {11:.4f},\n\
            acc_{0} - {12:.4f}, acc_{1} - {13:.4f}, acc_{2} - {14:.4f}, acc_{3} - {15:.4f}, acc_{4} - {16:.4f}, acc_avg - {17:.4f},\n\
            f1_{0} - {18:.4f}, f1_{1} - {19:.4f}, f1_{2} - {20:.4f}, f1_{3} - {21:.4f}, f1_{4} - {22:.4f}, f1_micro - {23:.4f}, f1_macro - {24:.4f},\n\
            thresh_{0} - {25:4f}, thresh_{1} - {26:4f}, thresh_{2} - {27:4f}, thresh_{3} - {28:4f}, thresh_{4} - {29:4f} =='.\
            format(hparams.id_to_class[0], hparams.id_to_class[1], hparams.id_to_class[2], hparams.id_to_class[3], hparams.id_to_class[4], auc[0], auc[1], auc[2], auc[3], auc[4], auc['micro'], auc['macro'], acc[0], acc[1], acc[2], acc[3], acc[4], acc['avg'],
            f1[0], f1[1], f1[2], f1[3], f1[4], f1['micro'], f1['macro'], best_thresh[0], best_thresh[1], best_thresh[2], best_thresh[3], best_thresh[4]))
    return auc['micro']
