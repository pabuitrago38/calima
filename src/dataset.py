from argparse import ArgumentParser
import logging
import numpy as np
import scipy.sparse
from scipy.linalg import norm
from glob import glob
import pprint
from operator import mul
from time import time
import torch
from torch.utils.data import Dataset

from prepareData import prepare


class Dataset(Dataset):
  def __init__(self, mode, in_data_file, ref_dataset=None, output_raw=False):
    self.output_raw = output_raw

    # Data can be prepared for 1) Time > 0 sec or 2) Time > 5min.
    self.mode = mode

    self.data, GroupCategories, PartitionCategories, ReqGRESCategories, \
        ReqMemTypeCategories, ReqGPUCategories, QOSCategories = prepare(in_data_file)
    if ref_dataset is None:
      self.GroupCategories = list(GroupCategories)
      self.PartitionCategories = list(PartitionCategories)
      self.ReqGRESCategories = list(ReqGRESCategories)
      self.ReqMemTypeCategories = list(ReqMemTypeCategories)
      self.ReqGPUCategories = list(ReqGPUCategories)
      self.QOSCategories = list(QOSCategories)
      self.dims = 162
    else:
      # Copy categories from the reference dataset, 
      # because the category groups may be different.
      self.GroupCategories = ref_dataset.GroupCategories
      self.PartitionCategories = ref_dataset.PartitionCategories
      self.ReqGRESCategories = ref_dataset.ReqGRESCategories
      self.ReqMemTypeCategories = ref_dataset.ReqMemTypeCategories
      self.ReqGPUCategories = ref_dataset.ReqGPUCategories
      self.QOSCategories = ref_dataset.QOSCategories
      self.dims = 162
      # Filter everything in groups which are not on ref_dataset.
      self.data = [x for x in self.data if
          x['Group'] in self.GroupCategories and
          x['Partition'] in self.PartitionCategories and
          x['ReqGRES'] in self.ReqGRESCategories and
          x['ReqMemType'] in self.ReqMemTypeCategories and
          x['ReqGPU'] in self.ReqGPUCategories and
          x['QOS'] in self.QOSCategories
      ]

    # To remove all point with RealWait == 0
    if self.mode == 'classify_5min':
      self.data = [x for x in self.data if x['RealWait'].total_seconds() >= 0]
    if self.mode == 'regression_5min':
      self.data = [x for x in self.data if x['RealWait'].total_seconds() > 60.0*5]

  def to_onehot(self, label, categories):
    num_labels = len(categories)
    label_onehot = np.zeros((num_labels,), dtype=float)
    np.put(label_onehot, categories.index(label) % num_labels, 1)
    return label_onehot.tolist()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data[index]

    raw = item.copy() # Copy so that we do not change the member object.
    raw['RealWait'] = raw['RealWait'].total_seconds()
    raw['EligibleWait'] = raw['EligibleWait'].total_seconds()

    # Make a feature, including transform to onehot vector.
    sample = []
    sample += self.to_onehot(raw['Group'], self.GroupCategories)
    sample += self.to_onehot(raw['Partition'], self.PartitionCategories)
    sample.append(raw['ReqCPUS'] / 896.)
    sample += self.to_onehot(raw['ReqGRES'], self.ReqGRESCategories)
    sample += self.to_onehot(raw['ReqMemType'], self.ReqMemTypeCategories)
    sample.append(raw['ReqMem'] / 12288000.)
    sample.append(raw['ReqNodes'] / 32.)
    sample += self.to_onehot(raw['ReqGPU'], self.ReqGPUCategories)
    sample.append(raw['Timelimit'] / 336.)
    sample += self.to_onehot(raw['QOS'], self.QOSCategories)

    # To train the classifier RealWait > 5 min.
    if self.mode == 'classify_5min':
      label = 1 if raw['RealWait'] > 60 * 5 else 0
    # To train the classifier RealWait > 0.
    elif self.mode == 'classify_0sec':
      label = 1 if raw['RealWait'] > 0 else 0
    elif self.mode == 'regression_5min':
      label = raw['RealWait']
    else:
      raise Exception('Not implemented.')

    if "regression" in self.mode:
      label = np.array([label], dtype=np.float32)
    else:
      label = np.array([label], dtype=int)

    if self.output_raw:
      return {'sample': np.array(sample, dtype=np.float32), 
              'label': label,
              'raw': raw,
            }
    else:
      return {'sample': np.array(sample, dtype=np.float32), 
              'label': label,
            }




if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument('--in_data_file', default='/Users/paola/Google Drive/000-Development/Calima/Data/rawData1-P-filled.txt')
  args = parser.parse_args()

  dataset = Dataset(args.in_data_file)
  print len(dataset)
  print dataset[1000]
