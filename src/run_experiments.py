import logging
import argparse
import os, os.path as op
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from nn_train import train, get_parser

if __name__ == "__main__":

  parser = get_parser()
  parser.add_argument('--experiments_dir', default='/tmp/calima_checkpoints/experiments')
  base_args = parser.parse_args()

  logging.basicConfig(level=base_args.logging_level, format='%(levelname)s: %(message)s')

  experiments = [
     # {'_exper':0, 'mode':'classify_0sec','epochs':50,\
     #     'lr':0.001, 'n_layers':2, 'n_hidden_units':500, \
     #     'batch_size':256, 'save_freq_epochs':1, 'optimizer':'Adam', \
     #     'weight_decay':0, 'train_space':'Log', 'eval_space':'Log', \
     #     'dataset_type':'PT', 'train_loss':'L2' },
     {'_exper':1, 'mode':'classify_0sec','epochs':50,\
         'lr':0.001, 'n_layers':1, 'n_hidden_units':500, \
         'batch_size':256, 'save_freq_epochs':1, 'optimizer':'Adam', \
         'weight_decay':0, 'train_space':'Log', 'eval_space':'Log', \
         'dataset_type':'PT', 'train_loss':'L2' }, 
     {'_exper':2, 'mode':'classify_0sec','epochs':50,\
         'lr':0.001, 'n_layers':3, 'n_hidden_units':500, \
         'batch_size':256, 'save_freq_epochs':1, 'optimizer':'Adam', \
         'weight_decay':0, 'train_space':'Log', 'eval_space':'Log', \
         'dataset_type':'PT', 'train_loss':'L2' },
     {'_exper':3, 'mode':'classify_0sec','epochs':50,\
         'lr':0.001, 'n_layers':4, 'n_hidden_units':500, \
         'batch_size':256, 'save_freq_epochs':1, 'optimizer':'Adam', \
         'weight_decay':0, 'train_space':'Log', 'eval_space':'Log', \
         'dataset_type':'PT', 'train_loss':'L2' },
     {'_exper':4, 'mode':'classify_0sec','epochs':50,\
         'lr':0.001, 'n_layers':5, 'n_hidden_units':500, \
         'batch_size':256, 'save_freq_epochs':1, 'optimizer':'Adam', \
         'weight_decay':0, 'train_space':'Log', 'eval_space':'Log', \
         'dataset_type':'PT', 'train_loss':'L2' },
     {'_exper':4, 'mode':'classify_0sec','epochs':50,\
         'lr':0.001, 'n_layers':6, 'n_hidden_units':500, \
         'batch_size':256, 'save_freq_epochs':1, 'optimizer':'Adam', \
         'weight_decay':0, 'train_space':'Log', 'eval_space':'Log', \
         'dataset_type':'PT', 'train_loss':'L2' },
    

    # {'_exper': 1, 'batch_size': 128, 'n_layers': 6, 'epochs': 10},
    #{'_exper': 2, 'dataset_type': 'KK', 'epochs': 20}


    # parser.add_argument('--save_freq_epochs', default=5, type=int)
    # parser.add_argument('--batch_size', default=256, type=int)
    # parser.add_argument('--n_layers', default=2, type=int)
    # parser.add_argument('--n_hidden_units', default=500, type=int)
    # parser.add_argument('--epochs', default=5000, type=int,
    #     help='number of epochs to train.')
    # parser.add_argument('--lr', default=0.00001, type=float,
    #     help='learning rate.')
    # parser.add_argument('--weight_decay', default=0.0, type=float,
    #     help='weight decay.')
    # parser.add_argument('--mode', default='regression_5min',choices=['classify_0sec', 'classify_5min', 'regression_5min'])
    # parser.add_argument('--train_loss', default='L1',choices=['L1', 'L2'])
    # parser.add_argument('--dataset_type', default='PT',choices=['PT', 'KK'])
    # parser.add_argument('--optimizer', default='Adam',choices=['Adam', 'SGD'])
    # parser.add_argument('--train_space', default='Log',choices=['Log', 'Real'])
    # parser.add_argument('--eval_space', default='Log',choices=['Log', 'Real'])

  ]

  for experiment in experiments:

    # Form and print the name of the expeeriment.
    name = '_'.join(['%s=%s' % (k, v) for k, v in sorted(experiment.iteritems())])
    print 'Starting experiment %s' % name

    # Add values from experiemnts to the base argparser.
    d = deepcopy(vars(base_args))
    d.update(experiment)

    # Set the value of 'experiment_dir'.
    experiment_dir = op.join(base_args.experiments_dir, '%s' % name)
    print 'Experiment_dir:', experiment_dir
    d['checkpoint_dir'] = experiment_dir

    # Create an argparse namespace, properly set up for the experiement.
    args = argparse.Namespace(**d)

    # Train.
    train(args)
