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


def ROC(loader, net, use_gpu=False, plot_file=None):
  Y_gt = np.empty((0,))
  Y_prob = np.empty((0,2))
  for batch in loader:

    # get the inputs
    features, labels = batch['sample'], batch['label']

    # wrap them in Variable
    if use_gpu:
      features, labels = Variable(features.cuda()), Variable(labels.cuda())
    else:
      features, labels = Variable(features), Variable(labels)
    labels = torch.squeeze(labels)

    outputs = net(features)
    outputs = torch.nn.functional.softmax(outputs, dim=1)

    Y_gt = np.hstack((Y_gt, labels.data))
    Y_prob = np.vstack((Y_prob, outputs.data))

  tpr = []
  fpr = []
  for k0 in np.arange(0., 1., 0.0001):
    k1 = 1. - k0
    Y_pred = Y_prob[:,0] * k0 < Y_prob[:,1] * k1
    num_pos = np.sum(Y_gt)
    num_neg = len(Y_gt) - num_pos
    tpr.append(np.count_nonzero(np.bitwise_and(Y_pred, Y_gt == 1)) / float(num_pos))
    fpr.append(np.count_nonzero(np.bitwise_and(Y_pred, Y_gt == 0)) / float(num_neg))

  Y_pred = Y_prob[:,0] < Y_prob[:,1]
  # print 'positives model', np.count_nonzero(Y_pred) / float(len(Y_pred))
  # print 'positives data', np.count_nonzero(Y_gt) / float(len(Y_gt))
  acc = np.count_nonzero(Y_pred == Y_gt) / float(len(Y_pred))
  area = -np.trapz(y=tpr, x=fpr)

  # Plot also the training points
  plt.plot(fpr, tpr, '-o', linewidth=1, markersize=5)
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  if plot_file is not None:
    plt.savefig(plot_file)

  return area, acc


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--in_data_file', default='/Users/paola/Google Drive/000-Development/Calima/Data/rawData1-P-filled.txt')
  parser.add_argument('--ref_data_file', default=None)
  parser.add_argument('--checkpoint_path', default='/tmp/calima_checkpoints/epoch_1000.pth')
  parser.add_argument('--out_plot_path', default='/Users/paola/Google Drive/000-Development/Calima/Graphs/roc.png')
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--logging_level', type=int, default=20, choices=[10,20,30,40],
      help='10 = debug (everything), 20 = info + warning and errors, 30 = warning + errors, 40 = error')
  args = parser.parse_args()

  logging.basicConfig(level=args.logging_level, format='%(levelname)s: %(message)s')

  # Data.
  if args.ref_data_file is not None:
    ref_dataset = Dataset(in_data_file=args.ref_data_file)
    dataset = Dataset(in_data_file=args.in_data_file, ref_dataset=ref_dataset)
  else:
    dataset = Dataset(in_data_file=args.in_data_file)
  loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

  # Model.
  net = ClassificationModel(input_nc=dataset.dims)
  load_network(args.checkpoint_path, net)

  area, acc = ROC(loader, net,  use_gpu=args.use_gpu, plot_file=args.out_plot_path)
  print 'accuracy at k1=k2', acc, 'area', area

