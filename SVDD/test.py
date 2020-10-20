import time
import scipy
import code
import os, torch, sys
import torch
import csv
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from skimage.util import random_noise
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from hparams import hparams
from data import ChestData
from model import Encoder, Decoder
from metric import accuracy_metrics

import code

epsilon = 0.0000000001

plt.switch_backend('agg')

def add_noise(images):
    if hparams.cuda:
        images = images.cpu()

    images = images.numpy()
    lst_noisy = []
    sigma = 0.155
    for i in range(images.shape[0]):
        noisy = random_noise(images[i], var=sigma ** 2)
        lst_noisy.append(noisy)
    return torch.Tensor(np.array(lst_noisy))


def test(model_path, send_stats=False, accuracy=False, data='valid', plot_name='valid'):
    if data == 'valid':
        test_dataset = ChestData(data_csv=hparams.valid_csv, data_dir=hparams.valid_dir,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.485), (0.229))
                            ]))
    else:
        test_dataset = ChestData(data_csv=hparams.test_csv, data_dir=hparams.test_dir,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.485), (0.229))
                            ]))

    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=2)


    if hparams.cuda:
        encoder = Encoder().cuda(hparams.gpu_device)
        checkpoint = torch.load(model_path, map_location=hparams.gpu_device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.center = checkpoint['center']
        encoder.radius = checkpoint['radius']
    else:
        encoder = Encoder()
        checkpoint = torch.load(model_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.center = checkpoint['center']
        encoder.radius = checkpoint['radius']

    # print('Model loaded')

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    print('Testing model on {0} examples. '.format(len(test_dataset)))
    encoder = encoder.eval()
    # print('Validating model on {0} examples. '.format(len(validation_loader)))
    # print(encoder.center)
    # print(encoder.radius)
    with torch.no_grad():
        scores_list = []
        labels_list = []
        test_loss = 0
        for (img, labels, _) in tqdm(test_loader):
            # img = Variable(img.float(), requires_grad=False)
            # labels = Variable(labels.float(), requires_grad=False)
            scores = None
            if hparams.cuda:
                img = img.cuda(hparams.gpu_device)
                labels = labels.cuda(hparams.gpu_device)

            z = encoder(img)

            dist = torch.sum((z - encoder.center) ** 2, dim=1)
            if hparams.objective == 'soft-boundary':
                scores = dist - encoder.radius ** 2
                test_loss += (1 / hparams.nu) * torch.sum(torch.max(torch.zeros_like(scores), scores))
            else:
                scores = dist
                test_loss += torch.sum(dist)

            scores_list.append(scores)
            labels_list.append(labels)

        scores = torch.cat(scores_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        if hparams.cuda:
            plot_labels = labels.cpu()
            plot_scores = scores.cpu()
        plot_labels = plot_labels.numpy()
        plot_scores = plot_scores.numpy()

        accuracy = -1
        best_thr = -1

        if plot_name:
            groups = ("inlier", "outlier")
            color= ['red' if l == 0 else 'green' for l in plot_labels]
            lbl_name = ['inlier' if l == 0 else 'outlier' for l in plot_labels]
            plt.scatter(plot_scores[plot_labels == 1], plot_labels[plot_labels == 1], color='red', edgecolors='none', label='outlier', marker='+')
            plt.scatter(plot_scores[plot_labels == 0], plot_labels[plot_labels == 0], color='green', edgecolors='none', label='inlier', marker='+')
            plt.title('Scores vs labels')
            plt.legend(loc=0)
            plt.savefig(hparams.result_dir+plot_name+'_scores.png')
            print('scores plot saved to {}'.format(hparams.result_dir+plot_name+'_scores.png'))

            # Compute ROC curve and ROC area for each class
            # plot_scores -= np.min(plot_scores)
            # plot_scores /= np.max(plot_scores)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(1):
                fpr[i], tpr[i], thresholds = sklearn.metrics.roc_curve(plot_labels, plot_scores)
                roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(plot_labels.ravel(), plot_scores.ravel())
            roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
            plt.figure()
            lw = 2
            plt.plot(fpr[0], tpr[0], color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example for {} set.'.format(plot_name))
            plt.legend(loc="lower right")

            plt.savefig(hparams.result_dir+plot_name+'_roc_curve.png')
            print('roc-auc curve saved to {}'.format(hparams.result_dir+plot_name+'_scores.png'))

            for thr in thresholds:
                pred_labels = 1.0*np.array(plot_scores >= thr)
                acc = np.mean(1.0*(pred_labels == plot_labels))
                if accuracy <= acc:
                    best_thr = thr
                    accuracy = acc

        test_loss /= len(test_dataset)
        test_loss += encoder.radius ** 2 if hparams.objective == 'soft-boundary' else 0

        if hparams.cuda:
            labels = labels.cpu()
            scores = scores.cpu()

        labels = labels.view(-1).numpy()
        scores = scores.view(-1).detach().numpy()

        auc = roc_auc_score(labels, scores)
        print('== Test on -- '+model_path+' == auc - {0:.4f}, test_loss - {1:.4f}, best_threshold - {2:.4f}, accuracy - {3:.4f} =='.format(auc, test_loss, best_thr, accuracy))
        return auc, test_loss
#
#
# def rec_test(model_path, send_stats=False, accuracy=False, data='valid', all_th=False):
#
#     if data == 'valid':
#         test_dataset = RetinopathyData(data_csv=hparams.valid_csv, data_dir=hparams.valid_dir,
#                         transform=transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                         ]))
#     else:
#         test_dataset = RetinopathyData(data_csv=hparams.test_csv, data_dir=hparams.test_dir,
#                         transform=transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                         ]))
#
#     test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size,
#                             shuffle=False, num_workers=0)
#
#
#     if hparams.cuda:
#         generator = Generator().cuda(hparams.gpu_device)
#         checkpoint = torch.load(model_path, map_location=hparams.gpu_device)
#         generator.load_state_dict(checkpoint['generator_state_dict'])
#     else:
#         generator = Generator()
#         checkpoint = torch.load(model_path, map_location='cpu')
#         generator.load_state_dict(checkpoint['generator_state_dict'])
#
#     generator = generator.eval()
#     discriminator = discriminator.eval()
#
#     print('Model loaded')
#
#     Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor
#
#     print('Testing model on {0} examples. '.format(len(test_dataset)))
#
#     with torch.no_grad():
#         with torch.no_grad():
#             labels_list = []
#             z_list = []
#             imgs_list = []
#             for _ in range(hparams.repeat_infer):
#                 for (img, labels, img_names) in test_loader:
#                     img = Variable(img.float(), requires_grad=False)
#                     labels = Variable(labels.float(), requires_grad=False)
#                     if hparams.cuda:
#                         img = img.cuda(hparams.gpu_device)
#                         labels = labels.cuda(hparams.gpu_device)
#                     save_image(img, 'test_img.png', normalize=True)
#                     z = generator(img)
#                     save_image(z, 'test_z.png', normalize=True)
#                     z_list.append(z)
#                     imgs_list.append(img)
#                     labels_list.append(labels)
#
#         imgs = torch.cat(imgs_list, dim=0)
#         z = torch.cat(z_list, dim=0)
#         labels = torch.cat(labels_list, dim=0)
#         rec_loss = (imgs-z)**2
#         rec_loss = torch.mean(rec_loss, dim=[1,2,3])
#
#         best_thresh = hparams.rec_thresh
#         best_acc = 0
#
#         if all_th:
#             th_list = []
#             acc_list = []
#             for th in range(0, 20000, 1):
#                 th = 1.0*th/10000.0
#                 pred = (rec_loss <= th).float()
#                 acc = torch.mean((labels == pred).float())
#                 th_list.append(th)
#                 acc_list.append(acc)
#                 if acc >= best_acc:
#                     best_acc = acc
#                     best_thresh = th
#                     print(th, acc.item())
#             plt.plot(th_list, acc_list)
#             plt.xlabel('threshold')
#             plt.ylabel('accuracy')
#             plt.savefig(hparams.results_dir+'rec_acc_vs_thresh.png')
#
#         pred_labels = (rec_loss <= best_thresh).float()
#         if hparams.cuda:
#             pred_labels = pred_labels.cpu()
#             labels = labels.cpu()
#         pred_labels = pred_labels.view(-1).numpy()
#         labels = labels.view(-1).numpy()
#         conf_mat = confusion_matrix(labels, pred_labels, labels=[0, 1])
#         print('confusion matrix - ', conf_mat)
#         recall = conf_mat[0][0] / (conf_mat[0][0]+conf_mat[0][1]+epsilon)
#         precision = conf_mat[0][0] / (conf_mat[0][0]+conf_mat[1][0]+epsilon)
#         f1 = 2*recall*precision/(recall+precision+epsilon)
#         acc = accuracy_score(labels, pred_labels)
#
#         print('== '+data+' -- '+model_path+' with reconstruction error criterion == acc - {0:.4f}, precision - {1:.4f}, recall - {2:.4f}, f1 - {3:.4f}, thresh - {4:.4f} =='.format(acc, precision, recall, f1, best_thresh))
#     return acc
#
