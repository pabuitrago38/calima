import logging
import argparse
import os, os.path as op
from copy import deepcopy
from nn_train import train, get_parser

if __name__ == "__main__":

  parser = get_parser()
  parser.add_argument('--experiments_dir', default='/tmp/calima_checkpoints/experiments')
  base_args = parser.parse_args()

  logging.basicConfig(level=base_args.logging_level, format='%(levelname)s: %(message)s')

  experiments = [
    {'batch_size': 128, 'n_layers': 6, 'epochs': 10},
    {'dataset_type': 'KK', 'epochs': 10}
  ]

  for experiment in experiments:

    # Form and print the name of the expeeriment.
    name = '_'.join(['%s=%s' % (k, v) for k, v in experiment.iteritems()])
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
