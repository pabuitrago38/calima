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
  def __init__(self, in_data_file, output_raw=False):
    self.output_raw = output_raw

    self.data, GroupCategories, PartitionCategories, ReqGRESCategories, ReqMemTypeCategories, ReqGPUCategories, QOSCategories = prepare(in_data_file)
    self.GroupCategories = list(GroupCategories)
    self.PartitionCategories = list(PartitionCategories)
    self.ReqGRESCategories = list(ReqGRESCategories)
    self.ReqMemTypeCategories = list(ReqMemTypeCategories)
    self.ReqGPUCategories = list(ReqGPUCategories)
    self.QOSCategories = list(QOSCategories)
    self.dims = 170

  def to_onehot(self, label, categories):
    num_labels = len(categories)
    label_onehot = np.zeros((num_labels,), dtype=float)
    np.put(label_onehot, categories.index(label) % num_labels, 1)
    return label_onehot.tolist()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data[index]

    # Log on the output.
    item['RealWait'] = item['RealWait'].total_seconds()
    item['EligibleWait'] = item['EligibleWait'].total_seconds()

    # Make a feature, including transform to onehot vector.
    sample = []
    sample += self.to_onehot(item['Group'], self.GroupCategories)
    sample += self.to_onehot(item['Partition'], self.PartitionCategories)
    sample.append(item['ReqCPUS'] / 896.)
    sample += self.to_onehot(item['ReqGRES'], self.ReqGRESCategories)
    sample += self.to_onehot(item['ReqMemType'], self.ReqMemTypeCategories)
    sample.append(item['ReqMem'] / 12288000.)
    sample.append(item['ReqNodes'] / 32.)
    sample += self.to_onehot(item['ReqGPU'], self.ReqGPUCategories)
    sample.append(item['Timelimit'] / 336.)
    sample += self.to_onehot(item['QOS'], self.QOSCategories)
    label = np.log(item['RealWait'] + 1)

    if self.output_raw:
      return {'sample': np.array(sample, dtype=np.float32), 
              'label': np.array([label], dtype=np.float32),
              'item': item,
            }
    else:
      return {'sample': np.array(sample, dtype=np.float32), 
              'label': np.array([label], dtype=np.float32),
            }




if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument('--in_data_file', default='../CalimaData/rawData1-P-filled.txt')
  args = parser.parse_args()

  dataset = Dataset(args.in_data_file)
  print len(dataset)
  print dataset[1000]
