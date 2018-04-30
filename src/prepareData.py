from argparse import ArgumentParser
from datetime import datetime
import numpy as np

def prepare(in_data_file):

  with open(in_data_file) as f:
    lines = f.read().splitlines()

  # Cut out the header.
  lines = lines[1:]

  GroupCategories = set()
  PartitionCategories = set()
  ReqGRESCategories = set()
  ReqMemTypeCategories = set()
  ReqGPUCategories = set()
  QOSCategories = set()

  data = []
  for line in lines:
    words = line.split('|')

    JobID = words[0]
    Group = words[1]
    Partition = words[2]
    ReqCPUS = int(words[3])
    ReqGRES = words[4]
    if ReqGRES == '':
      ReqGRES = 'None'
    ReqMem = words[5]
    ReqMemType = ReqMem[-2:]
    ReqMem = int(ReqMem[:-2])
    ReqNodes = int(words[6])
    ReqGPU = words[7].split('/')[-1]
    if ReqGPU[:3] != 'gpu':
      ReqGPU = '0'
    assert len(words[8]) > 0
    Timelimit = words[8].split(':')[0]
    if '-' in Timelimit:
      Timelimit = Timelimit.split('-')
      Timelimit = int(Timelimit[0]) * 24 + int(Timelimit[1])
    else:
      Timelimit = int(Timelimit)
    if Timelimit == 0:
      Timelimit = 1  # WARNING: everything less than one hour is made to be one hour.
    QOS = words[9]
    Submit = datetime.strptime(words[10], '%Y-%m-%dT%X')
    Eligible = words[11]
    if Eligible == 'Unknown':
      continue
    Eligible = datetime.strptime(Eligible, '%Y-%m-%dT%X')
    Start = words[12]
    if Start == 'Unknown':
      continue
    Start = datetime.strptime(Start, '%Y-%m-%dT%X')

    RealWait = Start - Submit
    EligibleWait = Start - Eligible

    if Submit < datetime.strptime('2018-04-12T00:00:00', '%Y-%m-%dT%X'):
      continue  # Dates before April.

    GroupCategories.add(Group)
    PartitionCategories.add(Partition)
    ReqGRESCategories.add(ReqGRES)
    ReqMemTypeCategories.add(ReqMemType)
    ReqGPUCategories.add(ReqGPU)
    QOSCategories.add(QOS)

    data.append({
      'JobID': JobID,
      'Group': Group,
      'Partition': Partition,
      'ReqCPUS': ReqCPUS,
      'ReqGRES': ReqGRES,
      'ReqMemType': ReqMemType,
      'ReqMem': ReqMem,
      'ReqNodes': ReqNodes,
      'ReqGPU': ReqGPU,
      'Timelimit': Timelimit,
      'QOS': QOS,
      'Submit': Submit,
      'Eligible': Eligible,
      'Start': Start,
      'RealWait': RealWait,
      'EligibleWait': EligibleWait,
    })

  return data, GroupCategories, PartitionCategories, ReqGRESCategories, ReqMemTypeCategories, ReqGPUCategories, QOSCategories



if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument('--in_data_file', default='../CalimaData/rawData1-P-filled.txt')
  parser.add_argument('--out_data_file', default='../CalimaData/rawData1-P-filled-human-readable.txt')
  args = parser.parse_args()

  data, GroupCategories, PartitionCategories, ReqGRESCategories, ReqMemTypeCategories, ReqGPUCategories, QOSCategories = prepare(args.in_data_file)

  print 'GroupCategories', len(GroupCategories), GroupCategories
  print 'PartitionCategories', len(PartitionCategories), PartitionCategories
  print 'ReqGRESCategories', len(ReqGRESCategories), ReqGRESCategories
  print 'ReqMemTypeCategories', len(ReqMemTypeCategories), ReqMemTypeCategories
  print 'ReqGPUCategories', len(ReqGPUCategories), ReqGPUCategories
  print 'QOSCategories', len(QOSCategories), QOSCategories

  print 'total', len(data)

  def printRange(name):
    print '"%s" range' % name, min([x[name] for x in data]), '-', max([x[name] for x in data])
  printRange('Start')
  printRange('ReqCPUS')
  printRange('ReqMem')
  printRange('ReqNodes')
  printRange('Timelimit')
  printRange('RealWait')
  print max([x['RealWait'] for x in data]).total_seconds()
  printRange('EligibleWait')


  # Print a human readable file with the data
  SEP = ' '
  with open(args.out_data_file, 'w') as f:

    # Print the names of features to file.
    features = ['Group', 'Partition','ReqCPUS', 'ReqGRES','ReqMemType','ReqMem','ReqNodes','ReqGPU','Timelimit','QOS','RealWait','EligibleWait']

    for feat in features:
      f.write('%s%s' % (feat, SEP))
    f.write('\n')

    # Print values.

    for item in data:
      
      # sample += self.to_onehot(item['Group'], self.GroupCategories)
      # sample += self.to_onehot(item['Partition'], self.PartitionCategories)
      # sample.append(item['ReqCPUS'] / 896.)
      # sample += self.to_onehot(item['ReqGRES'], self.ReqGRESCategories)
      # sample += self.to_onehot(item['ReqMemType'], self.ReqMemTypeCategories)
      # sample.append(item['ReqMem'] / 12288000.)
      # sample.append(item['ReqNodes'] / 32.)
      # sample += self.to_onehot(item['ReqGPU'], self.ReqGPUCategories)
      # sample.append(item['Timelimit'] / 336.)
      # sample += self.to_onehot(item['QOS'], self.QOSCategories)
      # label = np.log(item['RealWait'].total_seconds() + 1)
      # features = ['Group', 'Partition','ReqCPUS', 'ReqGRES','ReqMemType','ReqMem','ReqNodes','ReqGPU','Timelimit','QOS','RealWait','EligibleWait']

      f.write('%s%s' % (item['Group'], SEP))
      f.write('%s%s' % (item['Partition'], SEP))
      f.write('%s%s' % (item['ReqCPUS'] / 896., SEP))
      f.write('%s%s' % (item['ReqGRES'], SEP))
      f.write('%s%s' % (item['ReqMemType'], SEP))
      f.write('%s%s' % (item['ReqMem'] / 12288000., SEP))
      f.write('%s%s' % (item['ReqNodes'] / 32., SEP))
      f.write('%s%s' % (item['ReqGPU'], SEP))
      f.write('%s%s' % (item['Timelimit'] / 336., SEP))
      f.write('%s%s' % (item['QOS'], SEP))
      f.write('%s%s' % (np.log(item['RealWait'].total_seconds() + 1), SEP))
      f.write('%s%s' % (np.log(item['EligibleWait'].total_seconds() + 1), SEP))

      f.write('\n')
      
