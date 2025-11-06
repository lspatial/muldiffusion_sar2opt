import torch
from torch.nn import functional as F
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
import argparse
import os
import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from sympy.physics.vector import outer
import numpy as np
import torch.optim as optim
import time
import sys
import gc
import copy
import datetime
from mappingknow.trainhelp import rmsec, r2_score, pearson_correlation
from mappingknow.trainhelp import RobustSSIMMAELoss,RobustSSIMMSELoss
from mappingknow.preswinunet import preSwinUnet
from tqdm import tqdm
import pandas as pd
import warnings
from torchvision.utils import make_grid, save_image


def cov2time(start,end):
  duration = end - start
  hour = duration.seconds//3600
  minute = duration.seconds//60
  second = duration.seconds % 60
  res=str(hour)+':'+ str(minute)+':'+str(second)
  return res

class SwinUnet_Trainer(object):

    def __init__(self, config,inchannel,savepath):
        self.config = config
        self.savepath = savepath
        if not os.path.exists(savepath): 
            os.makedirs(savepath) 
        self.inchannel = inchannel

    def trainstep(self,base_lr,iter_num, max_iterations,train_loader,criterion, device):
        total_loss = 0.0
        for batch in tqdm(train_loader):
            sar = batch['sar'].to(device)
            optical = batch['optical'].to(device)
            masks_pred = self.model(sar)
            masks_pred = masks_pred
            loss = criterion(masks_pred, optical.float())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Learning rate adjustment
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            if isinstance(self.optimizer, optim.SGD):
               for param_group in self.optimizer.param_groups:
                  param_group['lr'] = lr_
            total_loss +=loss.detach().cpu().numpy()
        return total_loss/len(train_loader)

    def evaluate(self,epoch, dataloader,device,save_images=True,num_to_save=10):
        torch.cuda.empty_cache()
        masks_pred_all, true_masks_all = [], []
        for i, batch in enumerate(tqdm(dataloader)):
            sar = batch['sar'].to(device)
            optical = batch['optical'].to(device)
            sar.to(device)
            optical.to(device)
            masks_pred = self.model(sar)
            masks_pred = masks_pred.cpu().detach()
            true_masks = optical.cpu().detach()
            masks_pred_all.append(masks_pred)
            true_masks_all.append(true_masks)
            del optical, sar, true_masks, masks_pred
            torch.cuda.empty_cache()
        masks_pred_all = torch.cat(masks_pred_all)
        true_masks_all = torch.cat(true_masks_all)
        def denormalize(x):
            samples = ((x * 0.5 + 0.5)*255).clamp(0, 255).to(torch.uint8)
            #samples = samples.permute(0, 2, 3, 1)
            samples = samples.contiguous()
            return samples
          
        if save_images:
            # Select a subset of images to save
            
           n_samples = min(num_to_save, masks_pred_all.shape[0])
    
           for i in range(n_samples):
              # Get samples - keep/restore to BCHW format
              pred = masks_pred_all[i:i+1]  # Shape: (1, C, H, W)
              true = true_masks_all[i:i+1]  # Shape: (1, C, H, W)
              
              # Clean up the tensors - replace NaN and Inf values
              pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
              true = torch.nan_to_num(true, nan=0.0, posinf=1.0, neginf=0.0)
              
              # Normalize to [0, 1] range for save_image
              pred = pred * 0.5 + 0.5
              true = true * 0.5 + 0.5
              
              # Ensure values are in valid range
              pred = pred.clamp(0, 1)
              true = true.clamp(0, 1)
              
              # Make a grid with the pair
              comparison = torch.cat([pred, true], dim=0)
              grid = make_grid(comparison, nrow=2, padding=20, normalize=False)
              
              # Explicitly specify the normalization in save_image to avoid internal conversion issues
              save_image(grid, f'{self.savepath}/comparison_{epoch}_{i}.png', normalize=False) 
         
        rmse = rmsec(true_masks_all,masks_pred_all)
        r2= r2_score(true_masks_all,masks_pred_all)
        cor = pearson_correlation(true_masks_all,masks_pred_all)
        
        
        return r2,rmse,cor

    def train(self,optimizer,args,traindata_loader, testdata_loader,loss='MSE',
              pretrained_file=None,pretrained_base_file=None,device=None):
        start_time = time.time()
        if args.batch_size != 24 and args.batch_size % 6 == 0:
            args.base_lr *= args.batch_size / 24
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if pretrained_file is not None:
            self.model = torch.load(pretrained_file, map_location=device)
        else:
            self.model = preSwinUnet(self.config,inchannel=self.inchannel, num_classes=3,shortcut=True)
            if pretrained_base_file is not None:
                self.model.load_from(pretrained_base_file, device)
            self.model.to(device)
        if optimizer=='sgd':
            self.optimizer = optim.SGD(self.model.parameters2opt(mode='all'), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer=='adam':
            self.optimizer = optim.Adam(self.model.parameters2opt(mode='all'), lr=args.base_lr)

        if args.clip_value is not None:
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.requires_grad: p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))
        # Clip gradients by norm
        if args.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters2opt(mode='all'), max_norm=args.clip_norm)
        hist = None
        best_rmse = 1e6
        show_time = 5
        max_iterations = args.num_epoch * len(traindata_loader)
        iter_num = 0
        # Define the loss function
        if loss=='MSE':
            criterion = nn.MSELoss()
        elif loss=='SIMMMAE':
            criterion = RobustSSIMMAELoss()
        elif loss=='SIMMMSE':
            criterion = RobustSSIMMSELoss(0.95)            
        criterion.to(device)    
        for epoch in range(1, args.num_epoch + 1):
            aloss = self.trainstep(args.base_lr,iter_num, max_iterations, traindata_loader, criterion,device)
            # Learning rate adjustment
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            if isinstance(optimizer, optim.SGD):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            iter_num = iter_num + 1
            print('Epoch ', epoch,' loss:',aloss)
            # Evaluate metrics at specified intervals
            if epoch == 1 or (epoch % show_time) == 0:
                r2,rmse,cor = self.evaluate(epoch,testdata_loader,device)
                print('train loss:', aloss, ',test r2:', r2, ', test rmse:', rmse, ', correlation:',cor)
                ametrics = pd.DataFrame(
                    {'epoch': epoch, 'train_loss': aloss, 'test_r2': r2.cpu().item(),
                     'test_rmse': rmse.cpu().item(), 'test_cor': cor.cpu().item()},
                      index=[epoch])
                if hist is None:
                    hist = ametrics
                else:
                    hist = pd.concat([hist, ametrics], axis=0, ignore_index=True)
                
            if epoch >= 1 and best_rmse>rmse:
                # Log metrics and save best model
                best_rmse = rmse
                torch.save(self.model.state_dict(), self.savepath +'/model_statedict_best.pth')
                modelFl = self.savepath + '/model_statedict_best.tor'
                torch.save(self.model, modelFl)
                hist.to_csv(self.savepath + '/train_hist.csv', index=False)
            print('Best Testing RMSE: ', best_rmse)


