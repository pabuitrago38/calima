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

from nn_model import *


def nn_reg_evaluation(loader, net, use_gpu=False, plot_file=None, train_space='Log', eval_space='Log'):
  
  class L1NormaLoss(nn.Module):
    def __call__(self, input, target):
     abs_dif = torch.abs(input-target)
     abs_norm = abs_dif/torch.abs(target)
     sum_abs = torch.sum(abs_norm)
     return sum_abs

  criterion_nl1 = L1NormaLoss()
  #if loss == 'l1':
  #criterion_l1 = nn.L1Loss()
  #elif loss == 'l2':
  #criterion_l2 = nn.L2Loss()
  #elif loss == 'l1-norm':
  
    

  # Dataset loss
  ds_loss_nl1 = 0.
  #ds_loss_l1 = 0.
  #ds_loss_l2 = 0.
  ds_size = len(loader.dataset)
  
  for batch in loader:

    # get the inputs and features
    features, labels = batch['sample'], batch['label']

    # wrap them in Variable
    if use_gpu:
      features, labels = Variable(features.cuda()), Variable(labels.cuda())
    else:
      features, labels = Variable(features), Variable(labels)
   
    # Evaluating in the log space
    ##outputs = net(features)
    ##labels = torch.log(labels + 1.)

    outputs = net(features)

    if train_space == 'Log' and eval_space == "Log":
      labels = torch.log(labels + 1.)
    elif train_space == 'Real' and eval_space == "Real":
      pass
    elif train_space == 'Log' and eval_space == "Real":
      # Evaluating in the real space
      outputs = torch.exp(outputs)-1
    elif train_space == 'Real' and eval_space == "Log":
      # Evaluation in the log space
      outputs = torch.log(outputs + 1.)
      labels = torch.log(labels + 1.)
    
    ds_loss_nl1 = ds_loss_nl1 + criterion_nl1(outputs, labels)
    #ds_loss_l1 = ds_loss_l1 + criterion_l1(outputs, labels)
    #ds_loss_l2 = ds_loss_l2 + criterion_l2(outputs, labels)

    nl1_loss = ds_loss_nl1/ds_size
    #l1_loss = ds_loss_l1/ds_size
    #l2_loss = ds_loss_l2/ds_size

  return nl1_loss#, l1_loss, l2_loss

