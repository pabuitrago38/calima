import argparse
import os, os.path as op
import numpy as np
import logging
from time import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset
from nn_model import *
from roc import ROC
from nn_reg_evaluation import nn_reg_evaluation

PRINT_FREQ_SEC = 2
SAVE_FREQ_EPOCHS = 5

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_file', default='/Users/paola/Google Drive/000-Development/Calima/Data/rawData1-P-filled.txt')
parser.add_argument('--test_data_file', default='/Users/paola/Google Drive/000-Development/Calima/Data/rawData2-P-filled.txt')
parser.add_argument('--checkpoint_dir', default='/tmp/calima_checkpoints/tmp',
    help='where to save the model. If None, will not save.')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=5000, type=int,
    help='number of epochs to train.')
parser.add_argument('--lr', default=0.00001, type=float,
    help='learning rate.')
parser.add_argument('--mode', choices=['classify_0sec', 'classify_5min', 'regression_5min'])
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--logging_level', type=int, default=20, choices=[10,20,30,40],
    help='10 = debug (everything), 20 = info + warning and errors, 30 = warning + errors, 40 = error')
args = parser.parse_args()

logging.basicConfig(level=args.logging_level, format='%(levelname)s: %(message)s')

# Data.

trainset = Dataset(mode=args.mode, in_data_file=args.train_data_file)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)

testset = Dataset(mode=args.mode, in_data_file=args.test_data_file, ref_dataset=trainset)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, drop_last=False)

# Model.

net = Model(input_nc=trainset.dims, mode=args.mode)

# Training.

if args.mode == 'regression_5min':
  criterion = nn.L1Loss()
else:
  criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Create checkpoint_dir if it does not exist.
if not op.exists(args.checkpoint_dir):
  os.makedirs(args.checkpoint_dir)
loss_log_path = op.join(args.checkpoint_dir, 'loss.log')
with open(loss_log_path, 'w') as f:
  f.write('epoch iter train_loss\n')
eval_log_path = op.join(args.checkpoint_dir, 'eval.log')
with open(eval_log_path, 'w') as f:
  if args.mode == 'regression_5min':
    f.write('epoch train_loss test_loss\n')
  else:
    f.write('epoch train_acc train_auroc test_acc, test_auroc\n')

print('Started training.')
start = time()
totalIter = 0;

for epoch in range(args.epochs):  # loop over the dataset multiple times

  # model to training mode.
  net.train()

  for ibatch, batch in enumerate(trainloader):

    # get the inputs
    features, labels = batch['sample'], batch['label']

    # wrap them in Variable
    if args.use_gpu:
      features, labels = Variable(features.cuda()), Variable(labels.cuda())
    else:
      features, labels = Variable(features), Variable(labels)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    ## Calls forward function
    
    outputs = net(features)
    
    if args.mode == 'regression_5min':
      # Regression

      #outputs = torch.log(outputs + 1.)
      labels = torch.log(labels + 1.)
    else:
      # Classification
      labels = torch.squeeze(labels, dim=1)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    totalIter = totalIter + 1
    # Print statistics roughly every PRINT_FREQ samples
    if time() - start > PRINT_FREQ_SEC:
      print('[%d, %5d] loss: %f' % (epoch + 1, ibatch + 1, loss.data))
      start = time()

      # Write logs.
      with open(loss_log_path, 'a') as f:
        #epoch_frac = epoch + ibatch / float(args.batch_size)
        f.write('%d %d  %.2f\n' % (epoch +1, totalIter, loss.data))

    # The epoch ends here.

  if (epoch + 1) % SAVE_FREQ_EPOCHS == 0:

    if args.mode == 'regression_5min':
      #Regression evaluation
      training_loss = nn_reg_evaluation(trainloader, net, use_gpu=args.use_gpu)
      testing_loss = nn_reg_evaluation(testloader, net, use_gpu=args.use_gpu)

      #print 'epoch', epoch + 1
      #print 'training_loss: %.3f' % training_loss
      #print 'testing_loss: %.3f' % testing_loss

      with open(eval_log_path, 'a') as f:
        f.write('%d  %.3f %.3f\n' %
          (epoch + 1, training_loss, testing_loss))
    else:
      print 'epoch', epoch + 1
      trainarea, trainacc = ROC(trainloader, net, use_gpu=args.use_gpu)
      print 'training, area_under_roc: %.3f, accuracy_at_k1=k2: %.3f' % (trainarea, trainacc)
      testarea, testacc = ROC(testloader, net, use_gpu=args.use_gpu)
      print 'testing,  area_under_roc: %.3f, accuracy_at_k1=k2: %.3f' % (testarea, testacc)
      with open(eval_log_path, 'a') as f:
        f.write('%d  %.3f %.3f %.3f %.3f\n' % 
            (epoch + 1, trainacc, trainarea,testacc, testarea))




    if args.checkpoint_dir is not None:
      # Create checkpoint_dir directory if does not exist (will run on the first time).
      if not op.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
      # Save the network to a file in checkpoint_dir directory.
      save_network(args.checkpoint_dir, net, epoch + 1, use_gpu=args.use_gpu)
