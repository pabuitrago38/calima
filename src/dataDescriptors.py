import os.path as op
from dataset import Dataset
import numpy as np 
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def plotHist(name_x, categories_x=None, bins=None, use_ylog_scale=False):
  if categories_x is not None:
    # For categorical variable, indicdes of categories are taken.
    variable_x = np.array([categories_x.index(x) for x in data[name_x]])
    bins = np.arange(len(categories_x) + 1) - 0.5
  else:
    variable_x = np.array(data[name_x])
  if use_ylog_scale:
    plt.gca().set_yscale('log')
  plt.hist(variable_x, bins=bins)
  if categories_x is not None:
    plt.xticks(np.arange(len(categories_x)), categories_x, rotation=90)
  plt.xlabel(name_x)
  plt.ylabel('# events')
  log_suffix = '_log' if use_ylog_scale else ''
  plt.tight_layout()
  plt.savefig(op.join(args.out_plot_dir, '%s_histogram%s.png' % (name_x, log_suffix)))
  plt.clf()

def plotScatter(name_x, name_y, categories_x=None, use_ylog_scale=False):
  if categories_x is not None:
    variable_x = np.array([categories_x.index(x) for x in data[name_x]])
  else:
    variable_x = np.array(data[name_x])
  variable_y = np.array(data[name_y])
  if use_ylog_scale:
    plt.scatter(x=variable_x, y=variable_y + 1)  # Add 1 for correctness of log scale
    plt.gca().set_yscale('log')
  else:
    plt.scatter(x=variable_x, y=variable_y)
  if categories_x is not None:
    plt.xticks(np.arange(len(categories_x)), categories_x, rotation=90)
  plt.xlabel(name_x)
  plt.ylabel(name_y)
  plt.tight_layout()
  log_suffix = '_log' if use_ylog_scale else ''
  plt.savefig(op.join(args.out_plot_dir, '%s_vs_%s%s.png' % (name_x, name_y, log_suffix)))
  plt.clf()


if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument('--in_data_file', default='/Users/paola/Google Drive/000-Development/Calima/Data/rawData1-P-filled.txt')
  parser.add_argument('--out_plot_dir', default='/Users/paola/Google Drive/000-Development/Calima/Graphs')
  args = parser.parse_args()

  # Collect raw data from the dataset.
  dataset = Dataset(in_data_file=args.in_data_file, output_raw=True)
  list_of_dicts = [x['raw'] for x in dataset]
  #list_of_dicts = [x for x in list_of_dicts if x['RealWait'] != 0]
  # Convert from a list of dicts to a dict of lists.
  data = dict(zip(list_of_dicts[0], zip(*[d.values() for d in list_of_dicts])))

  use_ylog_scale = True
  plotHist('RealWait', bins=100, use_ylog_scale=False)
  plotHist('EligibleWait')
  plotHist('ReqCPUS')
  plotHist('Group', dataset.GroupCategories)
  plotHist('Partition', dataset.PartitionCategories)
  plotHist('ReqGRES', dataset.ReqGRESCategories)
  plotHist('ReqMemType', dataset.ReqMemTypeCategories)
  plotHist('ReqGPU', dataset.ReqGPUCategories)
  plotHist('QOS', dataset.QOSCategories)
  plotScatter('ReqCPUS', 'RealWait')
  plotScatter('ReqMem', 'RealWait')
  plotScatter('ReqNodes', 'RealWait')
  plotScatter('Timelimit', 'RealWait')
  plotScatter('Group', 'RealWait', dataset.GroupCategories)
  plotScatter('Partition', 'RealWait', dataset.PartitionCategories)
  plotScatter('ReqGRES', 'RealWait', dataset.ReqGRESCategories)
  plotScatter('ReqMemType', 'RealWait', dataset.ReqMemTypeCategories)
  plotScatter('ReqGPU', 'RealWait', dataset.ReqGPUCategories)
  plotScatter('QOS', 'RealWait', dataset.QOSCategories)
