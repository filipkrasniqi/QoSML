from columns import *

from os import listdir, scandir
from os.path import isfile, join,expanduser, isdir
import pandas as pd
from sklearn import preprocessing
import pathlib

import torch
import scipy.sparse as sp
import itertools
from dataset import Dataset
from torch.utils.data import DataLoader
from dataset_container import DatasetContainer

import math

'''
Extension of DatasetContainer for paper "Understanding ..."
(https://arxiv.org/pdf/1807.08652.pdf)
Class handling import of dataset from paper for network_10 with regimes, TDs.
Given R different rhos (aka intensity levels), D different distributions,
we have S = RxD simulations. Among them:
- TR_S = training size = (R-3) x D
- V_S = validation size = 3 x D
- TE_S = test size = 3 x D
PS: 2 because we want to test, for each TD, on 3 different intensities,
i.e., low, medium, high.
'''
class UnderstandingDataset(DatasetContainer):
    def __init__(self, **kwargs):
        'Initialize understanding dataset'
        super(UnderstandingDataset, self).__init__(**kwargs)

    '''
    Override
    '''
    def init_variables_dataset(self):
        self.intensities = []
        self.traffic_distributions = []

        for filename in self.all_filenames:
            if "dataScaleFree_" in filename:
                current_f = filename.split("/")[-1]
                current_i = int(current_f.split("_")[3])
                current_td = current_f.split("_")[-1].split(".")[0]
                if current_i not in self.intensities:
                    self.intensities.append(current_i)
                if current_td not in self.traffic_distributions:
                    self.traffic_distributions.append(current_td)

        self.intensities.sort()
        self.traffic_distributions.sort()

        self.test_i = [ self.intensities[3], self.intensities[len(self.intensities)-3], self.intensities[31]]
        self.test_f = ["dataScaleFree_{}_{}_{}_16000_{}.txt".format(self.num_nodes, self.num_nodes, test_f[0], test_f[1]) for test_f in itertools.product(self.test_i, self.traffic_distributions)]
        self.train_f = ["dataScaleFree_{}_{}_{}_16000_{}.txt".format(self.num_nodes, self.num_nodes, train_f[0], train_f[1]) for train_f in itertools.product(self.intensities, self.traffic_distributions) if "dataScaleFree_{}_{}_{}_16000_{}.txt".format(self.num_nodes, self.num_nodes, train_f[0], train_f[1]) not in self.test_f]

    '''
    Override
    '''
    def init_cachefiles(self):
        self.folder_cache = join(*[self.dir_datasets_env, self.cache_dir])
        super().init_cachefiles()

    '''
    Override
    '''
    def get_info_simulation(self):
        filename_example = self.all_filenames[0]
        df_example = pd.read_csv(filename_example, sep=" ", header=None, index_col=False)
        return df_example.shape[0], int(math.sqrt(int((df_example.shape[1] - 2) / 2)))

    '''
    Override
    '''
    def init_base_directory(self):
        self.dir_datasets = join(*[expanduser('~'), 'notebooks', 'datasets', 'understanding_nn'])
        self.dir_datasets_env = join(*[self.dir_datasets, 'pytorch', self.topology])+"/"
        self.all_filenames = [f.path for f in scandir(self.dir_datasets_env) if not f.is_dir()]

    '''
    Override
    '''
    def init_columns(self):
        super().init_columns()
        self.all_columns = self.traffic_cols + self.delay_cols + ["cumulative_drops"]

        self.input_columns = self.traffic_cols
        self.output_columns = self.delay_cols

        prefixes_cols_self_edges = ["{}_{}".format(i, i) for i in range(self.max_num_nodes)]
        self.tensor_input_columns = [col for col in self.input_columns if not any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_self_edges)]
        self.tensor_output_columns = self.delay_cols

    '''
    Override
    '''
    def getDataframeFromSimulation(self, path):
        return pd.read_csv(path, sep=" ", header=None, names=list(self.all_columns),index_col=False)

    '''
    Override
    '''
    def aggregateDataframes(self):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        if not isdir(self.folder_cache) or not isfile(self.path_train) or not isfile(self.path_test) or not isfile(self.path_train_X) or not isfile(self.path_train_y) or not isfile(self.path_test_X) or not isfile(self.path_test_y) or not self.generate_tensors:
            files_exist = self.read_from_cache and isdir(self.folder_cache) and isfile(self.path_train) and isfile(self.path_test)
            if files_exist:
                df_train, df_test = self.init_dataframes_from_cache()
            else:
                print("INFO: constructing training set")
                for i,f in enumerate(self.train_f):
                    path = self.dir_datasets_env+f
                    if isfile(path):
                        df_train = pd.concat([df_train, self.getDataframeFromSimulation(path)], ignore_index=True)
                        print("INFO: Reading. Dataset = {}/{}.".format(i+1, len(self.train_f), end='\r', flush=True))
                    else:
                        print("WARNING: train dataset not present: ".format(f))

                print("INFO: training is done!\nINFO: constructing test set")
                for i, f in enumerate(self.test_f):
                    path = self.dir_datasets_env+f
                    if isfile(path):
                        df_test = pd.concat([df_test, self.getDataframeFromSimulation(path)], ignore_index=True)
                        print("INFO: Reading. Dataset = {}/{}.".format(i+1, len(self.test_f), end='\r', flush=True))
                    else:
                        print("WARNING: test dataset not present: ".format(f))

                df_train.to_csv(path_or_buf=self.path_train, sep=' ',index=False, header=False)
                df_test.to_csv(path_or_buf=self.path_test, sep=' ',index=False, header=False)

        return df_train, df_test

    '''
    Override
    TODO test
    '''
    def get_dataframe_visualization(self):
        df_test_visualization = pd.DataFrame()
        intensities = np.array([])
        traffic_distributions = ["M"] # only considering Poisson distr for visualization. About intensities: considering all three of them
        test_f = ["dataScaleFree_{}_{}_{}_16000_{}.txt".format(self.num_nodes, self.num_nodes, test_f[0], test_f[1]) for test_f in itertools.product(self.test_i, traffic_distributions)]
        already_printed = False
        for f in test_f:
            intensity = int(f.split("_")[3])
            path = self.dir_datasets_env+f
            if isfile(path):
                df_current = pd.read_csv(path, sep=" ", header=None, names=list(self.all_columns),index_col=False)
                df_current_samples = df_current
                if not already_printed:
                    print("Shape of single dataframe: {}".format(df_current_samples.shape))
                    already_printed = True
                df_test_visualization = pd.concat([df_test_visualization, df_current_samples], ignore_index=True)
                intensities = np.append(intensities, np.repeat(intensity, df_current_samples.shape[0]))
            else:
                print("no {}".format(f))
        df_test_visualization["intensity"] = intensities
        return df_test_visualization
