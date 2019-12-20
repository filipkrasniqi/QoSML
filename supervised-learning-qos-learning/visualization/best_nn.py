from os import scandir

import sys
from os.path import join, expanduser
import numpy as np

arguments = sys.argv

dataset_type = "ns3"
model = "{}_{}".format(arguments[1], dataset_type)

dir_datasets = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', 'exported', 'crossvalidation'])
path_r2_scores = join(*[dir_datasets, model, "r2_test_scores.txt"])

with open(path_r2_scores, 'r') as f2:
    r2_scores = f2.read()[1:-1]
    r2_scores = np.fromstring(r2_scores, sep=',')
    sorted_r2_scores = np.sort(r2_scores)

idx_max = np.argmax(r2_scores)

print("Search {}, CV {}, R2 = {}".format(int(idx_max / 3), idx_max % 3, np.max(r2_scores)))