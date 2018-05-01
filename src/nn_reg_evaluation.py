import argparse
import os, os.path as op
import numpy as np
import logging
from time import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import Dataset
from nn_model import *


def nn_reg_evaluation(loader, net, use_gpu=False, plot_file=None, loss='l1-norm'):
  
  if loss == 'l1':
    criterion = nn.L1Loss()
  elif loss == 'l2':
    criterion = nn.L2Loss()
  elif loss == 'l1-norm':
    class L1NormaLoss(nn.Module):
      def __call__(self, input, target):
       abs_dif = torch.abs(input-target)
       abs_norm = abs_dif/torch.abs(target)
       sum_abs = torch.sum(abs_norm)
       return sum_abs

    criterion = L1NormaLoss()
    

  # Dataset loss
  ds_loss = 0.
  ds_size = len(loader.dataset)
  
  for batch in loader:

    # get the inputs and features
    features, labels = batch['sample'], batch['label']

    # wrap them in Variable
    if use_gpu:
      features, labels = Variable(features.cuda()), Variable(labels.cuda())
    else:
      features, labels = Variable(features), Variable(labels)
   
    outputs = net(features)
    #outputs = torch.log(net(features) + 1.)
    labels = torch.log(labels + 1.)
    
    ds_loss = ds_loss + criterion(outputs, labels)

  return ds_loss/ds_size

