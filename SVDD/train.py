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
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data.sampler import WeightedRandomSampler
from tensorboardX import SummaryWriter
from functools import reduce
import operator
import copy
from tqdm import tqdm
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skimage.util import random_noise
from sklearn.metrics import roc_auc_score

from hparams import hparams
from data import ChestData
from model import Encoder, Decoder
from metric import accuracy_metrics

plt.switch_backend('agg')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm2d') != -1:
    #     torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     torch.nn.init.constant_(m.bias.data, 0.0)

def plot_cf(cf):
    fig = plt.figure()
    df_cm = pd.DataFrame(cf, range(2), range(2))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    return fig


def init_center(encoder, train_loader):
    encoder.center = torch.zeros(hparams.latent_dim)
    if hparams.cuda:
        encoder.center = encoder.center.cuda(hparams.gpu_device)
    encoder.eval()
    n_samples = 0
    with torch.no_grad():
        for i, (imgs, _, _) in enumerate(train_loader):
            imgs = Variable(imgs.float(), requires_grad=False)
            if hparams.cuda:
                imgs = imgs.cuda(hparams.gpu_device)
            outputs = encoder(imgs)
            n_samples += outputs.shape[0]
            encoder.center += torch.sum(outputs, dim=0)
    encoder.center = encoder.center / n_samples
    encoder.center[(abs(encoder.center) < hparams.epsilon) & (encoder.center < 0)] = -hparams.epsilon
    encoder.center[(abs(encoder.center) < hparams.epsilon) & (encoder.center > 0)] = hparams.epsilon
    print('Center initialization done.')
    return encoder

def train(resume=False):

    it = 0

    writer = SummaryWriter('../runs/'+hparams.exp_name)

    for k in hparams.__dict__.keys():
        writer.add_text(str(k), str(hparams.__dict__[k]))

    train_dataset = ChestData(data_csv=hparams.train_csv, data_dir=hparams.train_dir,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.485), (0.229))
                        ]))

    validation_dataset = ChestData(data_csv=hparams.valid_csv, data_dir=hparams.valid_dir,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.485), (0.229))
                        ]))

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=0)

    validation_loader = DataLoader(validation_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=0)

    print('loaded train data of length : {}'.format(len(train_dataset)))

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    def validation(encoder_, decoder_=None, send_stats=False, epoch=0):
        encoder_ = encoder_.eval()
        if decoder_:
            decoder_ = decoder_.eval()
        # print('Validating model on {0} examples. '.format(len(validation_loader)))
        with torch.no_grad():
            scores_list = []
            labels_list = []
            val_loss = 0
            for (img, labels, imgs_names) in validation_loader:
                img = Variable(img.float(), requires_grad=False)
                labels = Variable(labels.float(), requires_grad=False)
                scores = None
                if hparams.cuda:
                    img = img.cuda(hparams.gpu_device)
                    labels = labels.cuda(hparams.gpu_device)

                z = encoder_(img)

                if decoder_:
                    outputs = decoder_(z)
                    scores = torch.sum((outputs - img) ** 2, dim=tuple(range(1, outputs.dim())))# (outputs - img) ** 2
                    # rec_loss = rec_loss.view(outputs.shape[0], -1)
                    # rec_loss = torch.sum(torch.sum(rec_loss, dim=1))
                    val_loss += torch.sum(scores)
                    save_image(img, 'tmp/img_{}.png'.format(epoch), normalize=True)
                    save_image(outputs, 'tmp/reconstructed_{}.png'.format(epoch), normalize=True)

                else:
                    dist = torch.sum((z - encoder.center) ** 2, dim=1)
                    if hparams.objective == 'soft-boundary':
                        scores = dist - encoder.radius ** 2
                        val_loss += (1 / hparams.nu) * torch.sum(torch.max(torch.zeros_like(scores), scores))
                    else:
                        scores = dist
                        val_loss += torch.sum(dist)

                scores_list.append(scores)
                labels_list.append(labels)

            scores = torch.cat(scores_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            val_loss /= len(validation_dataset)
            val_loss += encoder_.radius ** 2 if decoder_ and hparams.objective == 'soft-boundary' else 0

            if hparams.cuda:
                labels = labels.cpu()
                scores = scores.cpu()

            labels = labels.view(-1).numpy()
            scores = scores.view(-1).detach().numpy()

            auc = roc_auc_score(labels, scores)

        return auc, val_loss
    ### validation function ends.

    if hparams.cuda:
        encoder = Encoder().cuda(hparams.gpu_device)
        decoder = Decoder().cuda(hparams.gpu_device)
    else:
        encoder = Encoder()
        decoder = Decoder()

    params_count = 0
    for param in encoder.parameters():
        params_count += np.prod(param.size())
    for param in decoder.parameters():
        params_count += np.prod(param.size())
    print('Model has {0} trainable parameters'.format(params_count))

    if not hparams.load_model:
        encoder.apply(weights_init_normal)
        decoder.apply(weights_init_normal)

    optim_params = list(encoder.parameters())
    optimizer_train = optim.Adam(optim_params, lr=hparams.train_lr, weight_decay=hparams.weight_decay,
                       amsgrad=hparams.optimizer == 'amsgrad')

    if hparams.pretrain:
        optim_params += list(decoder.parameters())
        optimizer_pre = optim.Adam(optim_params, lr=hparams.pretrain_lr, weight_decay=hparams.ae_weight_decay,
                           amsgrad=hparams.optimizer == 'amsgrad')
        # scheduler_pre = ReduceLROnPlateau(optimizer_pre, mode='min', factor=0.5, patience=10, verbose=True, cooldown=20)
        scheduler_pre = MultiStepLR(optimizer_pre, milestones=hparams.lr_milestones, gamma=0.1)

    # scheduler_train = ReduceLROnPlateau(optimizer_train, mode='min', factor=0.5, patience=10, verbose=True, cooldown=20)
    scheduler_train = MultiStepLR(optimizer_train, milestones=hparams.lr_milestones, gamma=0.1)

    print('Starting training.. (log saved in:{})'.format(hparams.exp_name))
    start_time = time.time()

    mode = 'pretrain' if hparams.pretrain else 'train'
    best_valid_loss = 100000000000000000
    best_valid_auc = 0
    encoder = init_center(encoder, train_loader)

    # print(model)
    for epoch in range(hparams.num_epochs):
        if mode == 'pretrain' and epoch == hparams.pretrain_epoch:
            print('Pretraining done.')
            mode = 'train'
            best_valid_loss = 100000000000000000
            best_valid_auc = 0
            encoder = init_center(encoder, train_loader)
        for batch, (imgs, labels, _) in enumerate(train_loader):

            # imgs = Variable(imgs.float(), requires_grad=False)

            if hparams.cuda:
                imgs = imgs.cuda(hparams.gpu_device)

            if mode == 'pretrain':
                optimizer_pre.zero_grad()
                z = encoder(imgs)
                outputs = decoder(z)
                # print(torch.max(outputs), torch.mean(imgs), torch.min(outputs), torch.mean(imgs))
                scores = torch.sum((outputs - imgs) ** 2, dim=tuple(range(1, outputs.dim())))
                # print(scores)
                loss = torch.mean(scores)
                loss.backward()
                optimizer_pre.step()
                writer.add_scalar('pretrain_loss', loss.item(), global_step=batch+len(train_loader)*epoch)

            else:
                optimizer_train.zero_grad()

                z = encoder(imgs)
                dist = torch.sum((z - encoder.center) ** 2, dim=1)
                if hparams.objective == 'soft-boundary':
                    scores = dist - encoder.radius ** 2
                    loss = encoder.radius ** 2 + (1 / hparams.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                loss.backward()
                optimizer_train.step()

                if hparams.objective == 'soft-boundary' and epoch >= hparams.warmup_epochs:
                    R = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - hparams.nu)
                    encoder.radius = torch.tensor(R)
                    if hparams.cuda:
                        encoder.radius = encoder.radius.cuda(hparams.gpu_device)
                    writer.add_scalar('radius', encoder.radius.item(), global_step=batch+len(train_loader)*epoch)
                writer.add_scalar('train_loss', loss.item(), global_step=batch+len(train_loader)*epoch)

            # pred_labels = (scores >= hparams.thresh)

            # save_image(imgs, 'train_imgs.png')
            # save_image(noisy_imgs, 'train_noisy.png')
            # save_image(gen_imgs, 'train_z.png')


            if batch % hparams.print_interval == 0:
                print('[Epoch - {0:.1f}, batch - {1:.3f}, loss - {2:.6f}]'.\
                format(1.0*epoch, 100.0*batch/len(train_loader), loss.item()))

        if mode == 'pretrain':
            val_auc, rec_loss = validation(copy.deepcopy(encoder), copy.deepcopy(decoder), epoch=epoch)
        else:
            val_auc, val_loss = validation(copy.deepcopy(encoder), epoch=epoch)

        writer.add_scalar('val_auc', val_auc, global_step=epoch)

        if mode == 'pretrain':
            best_valid_auc = max(best_valid_auc, val_auc)
            scheduler_pre.step()
            writer.add_scalar('rec_loss', rec_loss, global_step=epoch)
            writer.add_scalar('pretrain_lr', optimizer_pre.param_groups[0]['lr'], global_step=epoch)
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_pre_state_dict': optimizer_pre.state_dict(),
                }, hparams.model+'.pre')
            if best_valid_loss >= rec_loss:
                best_valid_loss = rec_loss
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_pre_state_dict': optimizer_pre.state_dict(),
                    }, hparams.model+'.pre.best')
                print('best model on validation set saved.')
            print('[Epoch - {0:.1f} ---> rec_loss - {1:.4f}, current_lr - {2:.6f}, val_auc - {3:.4f}, best_valid_auc - {4:.4f}] - time - {5:.1f}'\
                .format(1.0*epoch, rec_loss, optimizer_pre.param_groups[0]['lr'], val_auc, best_valid_auc, time.time()-start_time))

        else:
            scheduler_train.step()
            writer.add_scalar('val_loss', val_loss, global_step=epoch)
            writer.add_scalar('train_lr', optimizer_train.param_groups[0]['lr'], global_step=epoch)
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'center': encoder.center,
                'radius': encoder.radius,
                'optimizer_train_state_dict': optimizer_train.state_dict(),
                }, hparams.model+'.train')
            if best_valid_loss >= val_loss:
                best_valid_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'center': encoder.center,
                    'radius': encoder.radius,
                    'optimizer_train_state_dict': optimizer_train.state_dict(),
                    }, hparams.model+'.train.best')
                print('best model on validation set saved.')
            if best_valid_auc <= val_auc:
                best_valid_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'center': encoder.center,
                    'radius': encoder.radius,
                    'optimizer_train_state_dict': optimizer_train.state_dict(),
                    }, hparams.model+'.train.auc')
                print('best model on validation set saved.')
            print('[Epoch - {0:.1f} ---> val_loss - {1:.4f}, current_lr - {2:.6f}, val_auc - {3:.4f}, best_valid_auc - {4:.4f}] - time - {5:.1f}'\
                .format(1.0*epoch, val_loss, optimizer_train.param_groups[0]['lr'], val_auc, best_valid_auc, time.time()-start_time))

        start_time = time.time()
