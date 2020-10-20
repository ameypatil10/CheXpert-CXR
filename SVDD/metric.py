import numpy as np
from hparams import hparams

epsilon = 0.0000000001

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


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


def accuracy_metrics(pred_labels, labels, pred_logits, print_stat=False):
    if hparams.cuda:
        pred_labels = pred_labels.cpu()
        labels = labels.cpu()
        pred_logits = pred_logits.cpu()

    pred_labels = pred_labels.view(-1).numpy()
    labels = labels.view(-1).numpy()
    pred_logits = pred_logits.view(-1).detach().numpy()

    auc = roc_auc_score(labels, pred_logits)
    conf_mat = confusion_matrix(labels, pred_labels, labels=[0, 1])
        
    recall = conf_mat[0][0] / (conf_mat[0][0]+conf_mat[0][1]+epsilon)
    precision = conf_mat[0][0] / (conf_mat[0][0]+conf_mat[1][0]+epsilon)
    f1 = 2*recall*precision/(recall+precision+epsilon)

    acc_score = accuracy_score(labels, pred_labels)

    temp = classification_report(labels, pred_labels)


    if print_stat:
        print('confusion matrix - ', conf_mat)

    return auc, precision, recall, f1, acc_score, conf_mat
