import argparse
import os, os.path as op
import numpy as np
import logging
from time import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from dataset import Dataset

PRINT_FREQ = 50

parser = argparse.ArgumentParser()
parser.add_argument('--in_data_file', default='../CalimaData/rawData1-P-filled.txt')
parser.add_argument('--checkpoint_dir', default='/tmp/calima_checkpoints',
    help='where to save the model. If None, will not save.')
parser.add_argument('--log_path', default='/tmp/calima_logs_elig',
    help='where to write the accuracy logs to. If None, will not log.')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=1000, type=int,
    help='number of epochs to train.')
parser.add_argument('--lr', default=0.00001, type=float,
    help='learning rate.')
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--logging_level', type=int, default=20, choices=[10,20,30,40],
    help='10 = debug (everything), 20 = info + warning and errors, 30 = warning + errors, 40 = error')
args = parser.parse_args()

logging.basicConfig(level=args.logging_level, format='%(levelname)s: %(message)s')


# Network.

class Model(nn.Module):

  def __init__(self, input_nc, ndf=1000, n_layers=2, gpu_ids=[]):
    super(Model, self).__init__()
    self.gpu_ids = gpu_ids

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
        nn.Linear(ndf, 1, bias=True),
    ]
    # Wrapping up the list of layers for pytorch magic.
    self.model = nn.Sequential(*sequence)

    # Xavier intialization
    def weights_init_xavier(m):
      classname = m.__class__.__name__
      if hasattr(m, 'weight'):
        if classname.find('Conv') != -1:
          nn.init.xavier_normal(m.weight.data, gain=1)
        elif classname.find('Linear') != -1:
          nn.init.xavier_normal(m.weight.data, gain=1)
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

# Data.

trainset = Dataset(
    in_data_file=args.in_data_file)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Model.

net = Model(input_nc=trainset.dims)

# Training.

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

print('Started training.')
with open(args.log_path, 'w') as f:
  f.write('epoch i   loss\n')

for epoch in range(args.epochs):  # loop over the dataset multiple times
  start = time()

  for i, data in enumerate(trainloader, 0):

    # get the inputs
    features, labels = data['sample'], data['label']
    labels = torch.squeeze(labels)

    # wrap them in Variable
    if args.use_gpu:
      features, labels = Variable(features.cuda()), Variable(labels.cuda())
    else:
      features, labels = Variable(features), Variable(labels)

    # model to training mode.
    net.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    ## Calls forward function
    outputs = net(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Print statistics roughly every PRINT_FREQ samples
    if (i + 1) % PRINT_FREQ == 0:
      print('[%d, %5d, %3.1f sec.] loss: %f' %
          (epoch + 1, i + 1, time() - start, loss.data[0]))
      start = time()

    # Write logs.
    with open(args.log_path, 'a') as f:
      f.write('%d %d   %.2f\n' % (epoch, i, loss.data[0]))

    # The epoch ends here.

  # Save the model at the end of the epoch.
  if args.checkpoint_dir is not None:
    # Create checkpoint_dir directory if does not exist (will run on the first time).
    if not op.exists(args.checkpoint_dir):
      os.makedirs(args.checkpoint_dir)
    # Save the network to a file in checkpoint_dir directory.
    save_network(args.checkpoint_dir, net, epoch + 1, use_gpu=args.use_gpu)

  # evaluate on train and test data roughly every epoch
  # acc_train = evaluate(net, trainloader, 'train', args.use_gpu, max_batches=100)
  # if args.log_path is not None:
  #   with open(args.log_path, 'a') as f:
  #     f.write('%d   %.2f %.2f %.2f    %.2f %.2f %.2f\n' % (epoch, 
  #              acc_train[0], acc_train[1], (acc_train[0] + acc_train[1]) / 2.))

