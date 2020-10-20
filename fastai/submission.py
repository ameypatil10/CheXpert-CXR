import os, sys
import torch
import numpy as np
import pandas as pd
from fastai.vision import Path, get_transforms, ImageDataBunch, cnn_learner, Learner, models, DatasetType
import torch.nn as nn
from resnext import resnext101_64x4d

BS = 16
IMG_SZ = 320
IMG_MEAN = torch.FloatTensor([0.5027, 0.5027, 0.5027])
IMG_STD = torch.FloatTensor([0.2915, 0.2915, 0.2915])

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

chexpert_folder = 'home1/amey/CheXpert-v1.0-downsampled'
folder_path = '/'
model_path = Path('/home/amey/LTTS/fastai/models/')

model_names = {0: 'fastai-densenet-320-u0-stage-2', 
               1: 'fastai-densenet-320-u1-stage-1',
               2: 'fastai-resnet-320-u0-stage-1',
               3: 'fastai-resnet-320-u1-stage-2',
               4: 'fastai-resnext-320-u1-stage-1', 
               5: 'fastai-vgg-320-u0-stage-2', 
               6: 'fastai-vgg-320-u1-stage-1', 
               7: 'fastai-densenet-CT-phase2-u1-stage-2'
              }

def ensemble_method(outputs, mode='avg'):
    for idx in range(1, len(outputs)):
        outputs[0] += outputs[idx]
    return 1.0*outputs[0] / len(outputs)
    

data_tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=None, max_zoom=1, max_lighting=None,
                          max_warp=0, p_affine=0, p_lighting=0)


def save_preds(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    try:
        df = df[['Study']]
    except:
        try:
            df = df[['Path']]
        except:
            raise ValueError('csv has no attribute for path/study.')
            
    for lbl in ALL_LBLS:
        df[lbl] = np.zeros(len(df))

    test = ImageDataBunch.from_df(path=folder_path, df=df, folder=chexpert_folder, seed=0, 
                               label_col=ALL_LBLS, suffix='', valid_pct=1, ds_tfms=data_tfms,
                               bs=BS, size=IMG_SZ)#.normalize([IMG_MEAN, IMG_STD])
    
    IDs, outputs = test.valid_ds.x.items, []
    
    learn = cnn_learner(test, models.densenet121, model_dir=model_path, pretrained=False)
    learn.load(model_names[0])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)
    
    learn.load(model_names[1])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)
    
    learn = cnn_learner(test, models.resnet152, model_dir=model_path, pretrained=False)
    learn.load(model_names[2])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)

    learn.load(model_names[3])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)
    
    model = resnext101_64x4d(pretrained=None)
    model.last_linear = nn.Sequential(nn.Linear(32768, 2048), 
                                      nn.ReLU(True),
                                      nn.Dropout(),
                                      nn.Linear(2048, 14))
    learn = Learner(test, model, model_dir=model_path)
    learn.load(model_names[4])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)
    
    learn = cnn_learner(test, models.vgg19_bn, model_dir=model_path, pretrained=False)
    learn.load(model_names[5])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)
    
    learn.load(model_names[6])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)
    
    learn = cnn_learner(test, models.densenet121, model_dir=model_path, pretrained=False)
    learn.load(model_names[7])
    output, y, _ = learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
    outputs.append(output)
    
    output = ensemble_method(outputs, mode='avg')
    if torch.cuda.is_available():
        output = output.cpu()
    output = output.numpy()
    
    df = pd.DataFrame({
        'Path': IDs, 
        EVAL_LBLS[0]: output[:,1], 
        EVAL_LBLS[1]: output[:,2], 
        EVAL_LBLS[2]: output[:,3], 
        EVAL_LBLS[3]: output[:,4], 
        EVAL_LBLS[4]: output[:,5]})
    
    df.to_csv(output_csv, index=False)
    print('submission saved.')


if __name__ == '__main__':
    save_preds(sys.argv[1], sys.argv[2])
    