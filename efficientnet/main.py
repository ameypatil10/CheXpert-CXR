import argparse
import os
import numpy as np
from train import *
from test import *
from hparams import hparams
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()

if args.mode == 'train':
    train()

if args.mode == 'valid':
    auc = test(['../model/{}model.best'.format(hparams.exp_name)], data=(hparams.valid_csv, hparams.valid_dir), pred_csv=None)
    print(auc)
    
#     auc = test(['../model/baseline-top5-u0/model.best', '../model/baseline-top5-u1/model.best', '../model/baseline-top5-uI/model.best'], data=(hparams.valid_csv, hparams.valid_dir))
#     print(auc)
   
  