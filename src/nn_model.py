import argparse
import os, os.path as op
import numpy as np
import logging
from time import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn


class Model(nn.Module):

  def __init__(self, input_nc, mode, ndf=500, n_layers=2, gpu_ids=[]):
    super(Model, self).__init__()
    self.gpu_ids = gpu_ids

    # Number of output layers is one for regression and 2 for classification.
    output_nc = 1 if "regression" in mode else 2

    # The architecture of the network
    sequence = \
    [ 
        nn.Linear(input_nc, ndf, bias=True),
        nn.Sigmoid(),
    ] + \
    [ 
        nn.Linear(ndf, ndf, bias=True),
        nn.Sigmoid(),
    ] * n_layers + \
    [ 
        nn.Linear(ndf, output_nc, bias=True),
    ]
    # Wrapping up the list of layers for pytorch magic.
    self.model = nn.Sequential(*sequence)

    # Xavier intialization
    def weights_init_xavier(m):
      classname = m.__class__.__name__
      if hasattr(m, 'weight'):
        if classname.find('Conv') != -1:
          nn.init.xavier_normal_(m.weight.data, gain=1)
        elif classname.find('Linear') != -1:
          nn.init.xavier_normal_(m.weight.data, gain=1)
    self.model.apply(weights_init_xavier)

  # Implementation of the forward pass algorithm.
  def forward(self, input):
    # Run in parallel if running on GPU
    if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    # Run sequentially
    else:
        return self.model(input)



def save_network(save_dir, network, epoch_label, use_gpu=False):
  assert op.exists(save_dir), 'Save dir does not exist: %s' % save_dir
  save_filename = 'epoch_%s.pth' % epoch_label
  save_path = op.join(save_dir, save_filename)
  torch.save(network.cpu().state_dict(), save_path)
  if use_gpu and torch.cuda.is_available():
    network.cuda()

def load_network(load_path, network):
  assert op.exists(load_path), 'Load path does not exist: %s' % load_path
  network.load_state_dict(torch.load(load_path))
