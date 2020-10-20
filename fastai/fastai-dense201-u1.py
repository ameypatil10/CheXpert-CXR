#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
import pandas as pd
import numpy as np
from itertools import cycle
from fastai.vision import *
from fastai.metrics import *

torch.manual_seed(121)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(121)


# In[3]:


train_csv = Path('/home/amey/LTTS/data/train-u1.csv')
valid_csv = Path('/home/amey/LTTS/data/valid.csv')

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

BS = 32
IMG_SZ = 224
IMG_MEAN = torch.FloatTensor([0.5027, 0.5027, 0.5027])
IMG_STD = torch.FloatTensor([0.2915, 0.2915, 0.2915])
GPU_IDS = [1,2]
torch.cuda.set_device(GPU_IDS[0])

EXP = 'fastai-densenet201-224-u1'
RES_DIR = 'results/'+EXP+'/'
os.makedirs('results/', exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# In[4]:


data_tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=None, max_zoom=1, max_lighting=None,
                          max_warp=0, p_affine=0, p_lighting=0)

train = ImageDataBunch.from_csv('/home1/amey/CheXpert-v1.0-downsampled', csv_labels=train_csv, folder='', 
                               label_col=ALL_LBLS, delimiter=',', suffix='', valid_pct=0, ds_tfms=data_tfms,
                               bs=BS, size=IMG_SZ).normalize([IMG_MEAN, IMG_STD])

valid = ImageDataBunch.from_csv('/home1/amey/CheXpert-v1.0-downsampled', csv_labels=valid_csv, folder='', 
                               label_col=ALL_LBLS, delimiter=',', suffix='', valid_pct=1, ds_tfms=data_tfms,
                               bs=BS, size=IMG_SZ).normalize([IMG_MEAN, IMG_STD])

data = DataBunch.create(train_ds=train.train_ds, valid_ds=valid.valid_ds, bs=BS)

data


# In[5]:


from sklearn.metrics import roc_curve, auc

class AUC(Callback):

    def __init__(self, num_cl=14, pick='micro', plot_auc=False, plot_title=EXP+' - validation AUC', plot_path=RES_DIR+'valid_ROC_AUC.png'):
        self.id_to_class = {
            0: 'No Finding',
            1: 'Cardiomegaly',
            2: 'Edema',
            3: 'Consolidation',
            4: 'Atelectasis',
            5: 'Pleural Effusion',
            6: 'Enlarged Cardiomediastinum',
            7: 'Lung Opacity',
            8: 'Lung Lesion',
            9: 'Pneumonia',
            10: 'Pneumothorax',
            11: 'Pleural Other',
            12: 'Fracture',
            13: 'Support Devices',
            'micro': 'micro',
            'macro': 'macro',
        }
        self.name = str(self.id_to_class[pick])+'-AUC'
        self.pick = pick
        self.num_cl = num_cl
        self.plot_path = plot_path
        self.plot_title = plot_title
        self.plot_auc = plot_auc
        
    
    def on_epoch_begin(self, **kwargs):
        self.outputs, self.targets = [], []
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        self.outputs.append(last_output)
        self.targets.append(last_target)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        self.outputs = torch.sigmoid(torch.cat(self.outputs)).cpu().detach().numpy()
        self.targets = torch.cat(self.targets).cpu().numpy()
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(self.num_cl):
          fpr[i], tpr[i], _ = roc_curve(self.targets[:, i], self.outputs[:, i])
          roc_auc[self.id_to_class[i]] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        self.targets, self.outputs = self.targets[:,1:6], self.outputs[:,1:6]
        fpr["micro"], tpr["micro"], _ = roc_curve(self.targets.ravel(), self.outputs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(1,6)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(1,6):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= 5

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        if self.plot_auc:
            # Plot all ROC curves
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro (auc- {0:0.2f})'.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=2)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro (auc- {0:0.2f})'.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=2)

            colors = cycle(['blue', 'green', 'red', 'cyan', 'yellow', 'magenta', 'black'])
            for i, color in zip(range(12), colors):
                if roc_auc[self.id_to_class[i]] > 0:
                    plt.plot(fpr[i], tpr[i], color=color, lw=1,
                         label='{0} (auc- {1:0.2f})'.format(self.id_to_class[i], roc_auc[self.id_to_class[i]]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('{}'.format(self.plot_title))
            plt.legend(loc="lower right")
            plt.savefig(self.plot_path)
            print('roc-auc curve saved to {}'.format(self.plot_path))
        
        return add_metrics(last_metrics, roc_auc[self.id_to_class[self.pick]])

acc_02 = partial(accuracy_thresh, thresh=0.4)


# In[6]:


learn = cnn_learner(data, models.densenet201, metrics=[acc_02, AUC(pick=1), AUC(pick=2), AUC(pick=3), AUC(pick=4), AUC(pick=5), AUC(pick='micro', plot_auc=True, plot_path=RES_DIR+'valid_ROC_AUC.png')])
learn.model = torch.nn.DataParallel(learn.model, device_ids=GPU_IDS)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-3))
learn.save(EXP+'stage-1')
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(3*1e-5,1e-4))
learn.save(EXP+'stage-2')


# In[ ]:


# learn.validate(valid.valid_dl)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


BS = 32
IMG_SZ = 320
IMG_MEAN = torch.FloatTensor([0.5027, 0.5027, 0.5027])
IMG_STD = torch.FloatTensor([0.2915, 0.2915, 0.2915])
GPU_IDS = [0]
torch.cuda.set_device(GPU_IDS[0])

EXP = 'fastai-densenet201-320-u1'
RES_DIR = 'results/'+EXP+'/'
os.makedirs('results/', exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

data_tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=None, max_zoom=1, max_lighting=None,
                          max_warp=0, p_affine=0, p_lighting=0)

train = ImageDataBunch.from_csv('/home1/amey/CheXpert-v1.0-downsampled', csv_labels=train_csv, folder='', 
                               label_col=ALL_LBLS, delimiter=',', suffix='', valid_pct=0, ds_tfms=data_tfms,
                               bs=BS, size=IMG_SZ).normalize([IMG_MEAN, IMG_STD])

valid = ImageDataBunch.from_csv('/home1/amey/CheXpert-v1.0-downsampled', csv_labels=valid_csv, folder='', 
                               label_col=ALL_LBLS, delimiter=',', suffix='', valid_pct=1, ds_tfms=data_tfms,
                               bs=BS, size=IMG_SZ).normalize([IMG_MEAN, IMG_STD])

data = DataBunch.create(train_ds=train.train_ds, valid_ds=valid.valid_ds, bs=BS)

data


# In[ ]:


learn = cnn_learner(data, models.densenet201, metrics=[acc_02, AUC(pick=1), AUC(pick=2), AUC(pick=3), AUC(pick=4), AUC(pick=5), AUC(pick='micro', plot_auc=True, plot_path=RES_DIR+'valid_ROC_AUC.png')])
learn.model = torch.nn.DataParallel(learn.model, device_ids=GPU_IDS)
learn.load('fastai-densenet201-224-u1stage-2')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-4,3*1e-4))
learn.save(EXP+'-stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-5,3*1e-5))
learn.save(EXP+'-stage-2')


# In[ ]:


learn = cnn_learner(data, models.densenet201, loss_func=nn.BCEWithLogitsLoss(), metrics=[acc_02, AUC(pick=1), AUC(pick=2), AUC(pick=3), AUC(pick=4), AUC(pick=5), AUC(pick='micro', plot_auc=True, plot_path=RES_DIR+'valid_ROC_AUC.png', plot_title=EXP+' - validation AUC')])
learn.load('fastai-densenet201-320-u1-stage-1')


# In[ ]:


learn.validate(data.valid_dl)


# In[ ]:


# learn.load('fastai-densenet201-320-u1-stage-1')
# preds,y,losses = learn.get_preds(with_loss=True)
# losses = losses.reshape(y.shape)
# interp = ClassificationInterpretation(learn, preds, y, losses)
# interp.plot_top_losses(9, figsize=(7,7))


# In[ ]:


# # losses = losses[:,0]
# interp = ClassificationInterpretation(learn, preds, y, losses)
# top_losses, top_idxs = interp.top_losses()
# # interp.plot_top_losses(9, figsize=(7,7))


# In[ ]:


# idx=233
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ind = np.arange(0, 14)
# ax.bar(ind-0.2, preds[top_idxs[idx]], width=0.4, color='b', label='pred')
# ax.bar(ind+0.2, y[top_idxs[idx]], width=0.4, color='g', label='y')
# ax.set_xticks(ind)
# ax.set_yticks([0,1])
# ax.set_xticklabels(ALL_LBLS)
# ax.legend()
# ax.set_title('Top loss case no: {0}, loss value = {1:.4f}'.format(idx+1, top_losses[idx]))
# plt.show()
# data.valid_ds.x[top_idxs[idx]]


# In[ ]:


# idx=233
# print(preds[top_idxs[idx]])
# print(y[top_idxs[idx]])
# print(top_losses[idx])
# print(nn.BCELoss()(y[top_idxs[idx]], preds[top_idxs[idx]]))


# In[ ]:




