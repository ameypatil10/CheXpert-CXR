import numpy as np
from hparams import hparams

epsilon = 0.0000000001

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix
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


def accuracy_metrics(labels, pred_logits, plot_auc=None, plot_path='temp', best_thresh=None):
    if hparams.cuda:
        labels = labels.cpu()
        pred_logits = pred_logits.cpu()

    labels = labels.numpy()
    pred_logits = pred_logits.detach().numpy()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresh = [[]]*hparams.num_classes
    for i in range(hparams.num_classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(labels[:, i], pred_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    eval_label_idx = hparams.eval_id_to_class.keys()
    label_mask = [idx in eval_label_idx for idx in range(hparams.num_classes)]
    main_labels = labels[:, label_mask]
    main_pred_logits = pred_logits[:, label_mask]
    fpr["micro"], tpr["micro"], _ = roc_curve(main_labels.ravel(), main_pred_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in eval_label_idx]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in eval_label_idx:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(eval_label_idx)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    if plot_auc:

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro (auc- {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro (auc- {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = cycle(['blue', 'green', 'red', 'cyan', 'yellow', 'magenta', 'black'])
        for i, color in zip(range(hparams.num_classes), colors):
            if roc_auc[i] > 0:
                plt.plot(fpr[i], tpr[i], color=color, lw=1,
                     label='{0} (auc- {1:0.2f})'.format(hparams.id_to_class[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('{}'.format(plot_auc))
        plt.legend(loc="lower right")
        plt.savefig(plot_path+'_roc_curve.png')
        print('roc-auc curve saved to {}'.format(plot_path+'_roc_curve.png'))

    acc = {}
    if best_thresh is None:
        best_thresh = [0]*hparams.num_classes

        for lbl in range(hparams.num_classes):
            acc[lbl] = 0
            for thr in thresh[lbl]:
                pred_labels1 = pred_logits[:, lbl] > thr
                labels1 = labels[:, lbl]
                temp_acc = accuracy_score(labels1, pred_labels1)
                if temp_acc >= acc[lbl]:
                    acc[lbl] = temp_acc
                    best_thresh[lbl] = thr

    # for lbl in range(hparams.num_classes):
    #     acc[lbl] = 0
    #     pred_labels1 = pred_logits[:, lbl] > best_thresh[lbl]
    #     labels1 = labels[:, lbl]
    #     acc[lbl] = accuracy_score(labels1, pred_labels1)

    eval_acc = {i:acc[i] for i in eval_label_idx}
    acc['avg'] = sum(eval_acc.values())/len(eval_label_idx)

    pred_labels = pred_logits >= best_thresh

    f1 = f1_score(labels, pred_labels, average=None)
    f1 = {idx: f1[idx] for idx in range(hparams.num_classes)}
    f1['micro'] = f1_score(labels[:, label_mask], pred_labels[:, label_mask], average='micro')
    f1['macro'] = f1_score(labels[:, label_mask], pred_labels[:, label_mask], average='macro')

    conf_mat = multilabel_confusion_matrix(labels, pred_labels, labels=range(hparams.num_classes))

    return roc_auc, f1, acc, conf_mat, best_thresh


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