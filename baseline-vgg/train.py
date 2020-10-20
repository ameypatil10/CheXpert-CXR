import time
import code
import os, torch
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
from tensorboardX import SummaryWriter
from functools import reduce
import operator
from tqdm import tqdm
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skimage.util import random_noise

from hparams import hparams
from data import ChestData
from model import Discriminator
from metric import accuracy_metrics
from jigsaw import Discriminator as Jigsaw

plt.switch_backend('agg')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def plot_cf(cf):
    fig = plt.figure()
    df_cm = pd.DataFrame(cf, range(2), range(2))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
    return fig

def train(resume_path=None, jigsaw_path=None):

    writer = SummaryWriter('../runs/'+hparams.exp_name)

    for k in hparams.__dict__.keys():
        writer.add_text(str(k), str(hparams.__dict__[k]))

    train_dataset = ChestData(data_csv=hparams.train_csv, data_dir=hparams.train_dir, augment=hparams.augment,
                        transform=transforms.Compose([
                            transforms.Resize(hparams.image_shape),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5027, 0.5027, 0.5027), (0.2915, 0.2915, 0.2915))
                        ]))

    validation_dataset = ChestData(data_csv=hparams.valid_csv, data_dir=hparams.valid_dir,
                        transform=transforms.Compose([
                            transforms.Resize(hparams.image_shape),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5027, 0.5027, 0.5027), (0.2915, 0.2915, 0.2915))
                        ]))

    # train_sampler = WeightedRandomSampler()

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=2)

    validation_loader = DataLoader(validation_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=2)

    print('loaded train data of length : {}'.format(len(train_dataset)))

    adversarial_loss = torch.nn.BCELoss().to(hparams.gpu_device)
    discriminator = Discriminator().to(hparams.gpu_device)

    if hparams.cuda:
        discriminator = nn.DataParallel(discriminator, device_ids=hparams.device_ids)

    params_count = 0
    for param in discriminator.parameters():
        params_count += np.prod(param.size())
    print('Model has {0} trainable parameters'.format(params_count))

    if not hparams.pretrained:
#         discriminator.apply(weights_init_normal)
        pass
    if jigsaw_path:
        jigsaw = Jigsaw().to(hparams.gpu_device)
        if hparams.cuda:
            jigsaw = nn.DataParallel(jigsaw, device_ids=hparams.device_ids)
        checkpoints = torch.load(jigsaw_path, map_location=hparams.gpu_device)
        jigsaw.load_state_dict(checkpoints['discriminator_state_dict'])
        discriminator.module.model.features = jigsaw.module.feature.features
        print('loaded pretrained feature extractor from {} ..'.format(jigsaw_path))

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hparams.learning_rate, betas=(0.9, 0.999))

    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=0, verbose=True, cooldown=0)

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    def validation(discriminator, send_stats=False, epoch=0):
        print('Validating model on {0} examples. '.format(len(validation_dataset)))
        discriminator_ = discriminator.eval()

        with torch.no_grad():
            pred_logits_list = []
            labels_list = []

            for (img, labels, imgs_names) in tqdm(validation_loader):
                img = Variable(img.float(), requires_grad=False)
                labels = Variable(labels.float(), requires_grad=False)

                img_ = img.to(hparams.gpu_device)
                labels = labels.to(hparams.gpu_device)
                
                pred_logits = discriminator_(img_)

                pred_logits_list.append(pred_logits)
                labels_list.append(labels)

            pred_logits = torch.cat(pred_logits_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            val_loss = adversarial_loss(pred_logits, labels)

        return accuracy_metrics(labels.long(), pred_logits), val_loss

    print('Starting training.. (log saved in:{})'.format(hparams.exp_name))
    start_time = time.time()
    best_valid_auc = 0

    # print(model)
    for epoch in range(hparams.num_epochs):
        for batch, (imgs, labels, imgs_name) in enumerate(tqdm(train_loader)):

            imgs = Variable(imgs.float(), requires_grad=False)
            labels = Variable(labels.float(), requires_grad=False)

            imgs_ = imgs.to(hparams.gpu_device)
            labels = labels.to(hparams.gpu_device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_logits = discriminator(imgs_)

            d_loss = adversarial_loss(pred_logits, labels)

            d_loss.backward()
            optimizer_D.step()

            writer.add_scalar('d_loss', d_loss.item(), global_step=batch+epoch*len(train_loader))

            pred_labels = (pred_logits >= hparams.thresh)
            pred_labels = pred_labels.float()

            # if batch % hparams.print_interval == 0:
            #     auc, f1, acc, _, _ = accuracy_metrics(pred_labels, labels.long(), pred_logits)
            #     print('[Epoch - {0:.1f}, batch - {1:.3f}, d_loss - {2:.6f}, acc - {3:.4f}, f1 - {4:.5f}, auc - {5:.4f}]'.\
            #     format(1.0*epoch, 100.0*batch/len(train_loader), d_loss.item(), acc['avg'], f1[hparams.avg_mode], auc[hparams.avg_mode]))
        (val_auc, val_f1, val_acc, val_conf_mat, best_thresh), val_loss = validation(discriminator, epoch=epoch)

        for lbl in range(hparams.num_classes):
            fig = plot_cf(val_conf_mat[lbl])
            writer.add_figure('val_conf_{}'.format(hparams.id_to_class[lbl]), fig, global_step=epoch)
            plt.close(fig)
            writer.add_scalar('val_f1_{}'.format(hparams.id_to_class[lbl]), val_f1[lbl], global_step=epoch)
            writer.add_scalar('val_auc_{}'.format(hparams.id_to_class[lbl]), val_auc[lbl], global_step=epoch)
            writer.add_scalar('val_acc_{}'.format(hparams.id_to_class[lbl]), val_acc[lbl], global_step=epoch)
        writer.add_scalar('val_f1_{}'.format('micro'), val_f1['micro'], global_step=epoch)
        writer.add_scalar('val_auc_{}'.format('micro'), val_auc['micro'], global_step=epoch)
        writer.add_scalar('val_f1_{}'.format('macro'), val_f1['macro'], global_step=epoch)
        writer.add_scalar('val_auc_{}'.format('macro'), val_auc['macro'], global_step=epoch)
        writer.add_scalar('val_loss', val_loss, global_step=epoch)
        writer.add_scalar('val_f1', val_f1[hparams.avg_mode], global_step=epoch)
        writer.add_scalar('val_auc', val_auc[hparams.avg_mode], global_step=epoch)
        writer.add_scalar('val_acc', val_acc['avg'], global_step=epoch)
        scheduler_D.step(val_loss)
        writer.add_scalar('learning_rate', optimizer_D.param_groups[0]['lr'], global_step=epoch)

        torch.save({
            'epoch': epoch,
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, hparams.model+'.'+str(epoch))
        if best_valid_auc <= val_auc[hparams.avg_mode]:
            best_valid_auc = val_auc[hparams.avg_mode]
            for lbl in range(hparams.num_classes):
                fig = plot_cf(val_conf_mat[lbl])
                writer.add_figure('best_val_conf_{}'.format(hparams.id_to_class[lbl]), fig, global_step=epoch)
                plt.close(fig)
            torch.save({
                'epoch': epoch,
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                }, hparams.model+'.best')
            print('best model on validation set saved.')
        print('[Epoch - {0:.1f} ---> val_auc - {1:.4f}, current_lr - {2:.6f}, val_loss - {3:.4f}, best_val_auc - {4:.4f}, val_acc - {5:.4f}, val_f1 - {6:.4f}] - time - {7:.1f}'\
            .format(1.0*epoch, val_auc[hparams.avg_mode], optimizer_D.param_groups[0]['lr'], val_loss, best_valid_auc, val_acc['avg'], val_f1[hparams.avg_mode], time.time()-start_time))
        start_time = time.time()
