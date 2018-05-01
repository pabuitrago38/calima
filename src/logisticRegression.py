import argparse
import os, os.path as op
import numpy as np
import logging
from time import time
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import scipy.misc

from dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--in_data_file', default='../CalimaData/rawData1-P-filled.txt')
# parser.add_argument('--checkpoint_dir', default='/tmp/calima_checkpoints',
#     help='where to save the model. If None, will not save.')
# parser.add_argument('--log_path', default='/tmp/calima_logs_elig',
#     help='where to write the accuracy logs to. If None, will not log.')
# parser.add_argument('--batch_size', default=256, type=int)
# parser.add_argument('--epochs', default=1000, type=int,
#     help='number of epochs to train.')
# parser.add_argument('--lr', default=0.00001, type=float,
#     help='learning rate.')
# parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--logging_level', type=int, default=20, choices=[10,20,30,40],
    help='10 = debug (everything), 20 = info + warning and errors, 30 = warning + errors, 40 = error')
args = parser.parse_args()

logging.basicConfig(level=args.logging_level, format='%(levelname)s: %(message)s')


trainset = Dataset(in_data_file=args.in_data_file, output_raw=True)
X, Y = zip(*[(x['sample'], x['label'][0]) for x in trainset if x['label'][0] != 0])
X = np.array(list(X))
Y = np.array(list(Y))
print X.shape, Y.shape

# Target classes
for iy in range(len(Y)):
#  if Y[iy] == 0:
#    Y[iy] = 0
  if Y[iy] < 60 * 10:
    Y[iy] = 0
  else:
    Y[iy] = 1
print np.count_nonzero(Y), len(Y) - np.count_nonzero(Y)

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Training log probabilities.
Y_prob = logreg.predict_log_proba(X)
# Softmax
def softmax(Y):
  sums = np.sum(np.exp(Y), axis=1, keepdims=True)
  sums = np.hstack((sums, sums))
  return np.exp(Y) / sums
Y_prob = softmax(Y_prob)

tpr = []
fpr = []
for i, k1 in enumerate(np.arange(0., 1., 0.01)):
  k2 = 1. - k1
  Y_pred = (Y_prob[:,0] * k1 < Y_prob[:,1] * k2)
#  print np.hstack((Y_prob[:,0:1] * k1, Y_prob[:,1:2] * k2))[0], Y[0]
  num_pos = np.sum(Y_pred.astype(float))
  tpr.append(np.count_nonzero(np.bitwise_and(Y_pred == 1, Y == 1)) / num_pos)
  fpr.append(np.count_nonzero(np.bitwise_and(Y_pred == 1, Y == 0)) / num_pos)

print tpr, fpr

# Plot also the training points
plt.plot(fpr, tpr, 'o', linewidth=1, markersize=5)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
