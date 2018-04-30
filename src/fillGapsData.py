from argparse import ArgumentParser
from datetime import datetime

if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument('--in_data_file', default='../CalimaData/rawData1-P.txt')
  parser.add_argument('--out_data_file', default='../CalimaData/rawData1-P-filled.txt')
  args = parser.parse_args()

  with open(args.in_data_file) as f:
    lines = f.read().splitlines()

  parent_words = None

  for iline, line in enumerate(lines):
    words = line.split('|')

    if '.' in words[0]:
      assert parent_words[0] in words[0], 'The job id is not different'
      for iword, word in enumerate(words):
        if not words[iword]:
          words[iword] = parent_words[iword]
      lines[iline] = '|'.join(words)
    else:
      parent_words = list(words)

  with open(args.out_data_file, 'w') as f:
    for line in lines:
      f.write(line + '\n')


