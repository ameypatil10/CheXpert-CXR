import numpy as np
from hparams import hparams

epsilon = 0.0000000001

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy import interp

plt.switch_backend('agg')

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def Cmatrix(rater_a, rater_b, min_rating=0, max_rating=1):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat



def histogram(ratings, min_rating=0, max_rating=1):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def accuracy_metrics(labels, pred_labels, plot_auc=None, plot_path='temp', best_thresh=None):
    if hparams.cuda:
        labels = labels.cpu()
        pred_labels = pred_labels.cpu()

    labels = labels.numpy()
    pred_labels = pred_labels.detach().numpy()
    
    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, average='micro')
    
    conf_mat = None#confusion_matrix(labels, pred_labels, labels=list(range(hparams.num_classes)))

    return f1, acc, conf_mat


# import torch

# batch_sz = 128
# num_classes = 5

# pred_logits = torch.randn((batch_sz, num_classes))
# labels = torch.randint(0,2,(batch_sz,num_classes))

# print(pred_logits.shape)
# print(labels.shape)

# auc, f1, acc, conf_mat, best_thresh = accuracy_metrics(labels, pred_logits, plot_auc='temp')

# print(auc)
# print(f1)
# print(acc)
# print(best_thresh)
# print(conf_mat)