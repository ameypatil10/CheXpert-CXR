import argparse
import os
from train import *
from test import *
from hparams import hparams
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--thresh', type=float, default=0.5)
parser.add_argument('--rec_thresh', type=float, default=0.5)
parser.add_argument('--plot_scores', type=bool, default=True)

args = parser.parse_args()

if args.mode == 'train':
    train()
#
# if args.mode == 'valid':
#     best_thresh = 0
#     best_acc = 0
#     th_list = []
#     acc_list = []
#     for th in range(0, 100):
#         hparams.thresh = 1.0*th/100
#         print('-'*20, ' thresh = ', hparams.thresh, ' ', '-'*20)
#         acc = test(hparams.model+'.best', accuracy=True)
#         th_list.append(hparams.thresh)
#         acc_list.append(acc)
#         if(acc > best_acc):
#             best_thresh = hparams.thresh
#             best_acc = acc
#     hparams.thresh = best_thresh
#     acc = test(hparams.model+'.best')
#     plt.plot(th_list, acc_list)
#     plt.xlabel('threshold')
#     plt.ylabel('accuracy')
#     plt.savefig(hparams.results_dir+'acc_vs_thresh.png')
#     print('Best accuracy = ', best_acc, ' with threshold = ', best_thresh)
#
if args.mode == 'test':
    hparams.thresh = args.thresh
    test(hparams.model+'.train.auc', data='test', plot_name='test')

if args.mode == 'valid_test':
    hparams.thresh = args.thresh
    test(hparams.model+'.train.auc', data='valid', plot_name='valid')
#
# if args.mode == 'model':
#     best_auc = 0
#     model = 'no model'
#     try:
#         for i in range(200):
#             auc = test(hparams.model+'.'+str(i))
#             if auc >= best_auc:
#                 best_auc = auc
#                 model = hparams.model+'.'+str(i)
#     except:
#         print('best model = '+model+' with test auc = '+str(best_auc))
#
# if args.mode == 'rec_thresh':
#     rec_test(hparams.model+'.best', data=hparams.mnist_valid, all_th=True)
#
# if args.mode == 'rec_test':
#     hparams.rec_thresh = args.rec_thresh
#     rec_test(hparams.model+'.best', data=hparams.mnist_test)
