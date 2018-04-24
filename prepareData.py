from argparse import ArgumentParser
from datetime import datetime

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
    ReqMem = words[5]
    ReqMemType = ReqMem[-2:]
    ReqMem = int(ReqMem[:-2])
    ReqNodes = int(words[6])
    ReqGPU = words[7].split('/')[-1]
    if ReqGPU[:3] != 'gpu':
      ReqGPU = ''
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

    if Submit < datetime.strptime('2018-04-01T00:00:00', '%Y-%m-%dT%X'):
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
