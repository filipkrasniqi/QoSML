from columns import *

from os import listdir, scandir
from os.path import isfile, join,expanduser, isdir
import pandas as pd
from sklearn import preprocessing
import pathlib
from sklearn.model_selection import train_test_split

import torch
import scipy.sparse as sp
import itertools
from dataset_container import DatasetContainer
import math

'''
Extension of DatasetContainer for paper "Unveiling ...", aka Routenet algorithm
(https://github.com/knowledgedefinednetworking/Unveiling-the-potential-of-GNN-for-network-modeling-and-optimization-in-SDN/)
Split into test and train is made by taking all the dataset, shuffle it and sample it.
'''
class RoutenetDataset(DatasetContainer):
    def __init__(self,
               **kwargs):
        'Initialize routenet dataset'
        super(RoutenetDataset, self).__init__(**kwargs)

    '''
    Override
    '''
    def init_base_directory(self):
        self.dir_datasets = join(*[expanduser('~'), 'notebooks', 'datasets', 'routenet', self.topology, 'datasets'])
        self.dir_datasets_env = self.dir_datasets
        self.simulation_filename = "simulationResults.txt"

    '''
    Override
    '''
    def init_cachefiles(self):
        self.folder_cache = join(*[self.dir_datasets, self.cache_dir])
        super().init_cachefiles()

    '''
    Override
    '''
    def init_columns(self):
        super().init_columns()
        self.all_columns = self.get_all_columns_file()

        self.input_columns = self.traffic_cols
        self.output_columns = self.delay_cols

        prefixes_cols_self_edges = ["{}_{}".format(i, i) for i in range(self.max_num_nodes)]
        self.tensor_input_columns = [col for col in self.input_columns if not any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_self_edges)]
        self.tensor_output_columns = self.delay_cols

    '''
    Override
    '''
    def get_info_simulation(self):
        df_example = pd.read_csv(join(*[self.dir_datasets, np.array([dir for dir in listdir(self.dir_datasets) if "results" in dir])[0], self.simulation_filename]), sep=",", header=None, index_col=False)
        return df_example.shape[0], int(math.sqrt(int((df_example.shape[1] - 1) / 10)))

    '''
    Override
    '''
    def getDataframeFromSimulation(self, path):
        return pd.read_csv(path, sep=",", header=None, names=list(self.all_columns),index_col=False)

    '''
    Function to return the columns naming for raw data
    '''
    def get_all_columns_file(self):
        all_columns_file = []
        for i in range(self.num_nodes ** 2):
            src = int(i / self.num_nodes)
            dest = int(i % self.num_nodes)
            all_columns_file.append("traffic_{}_{}".format(src, dest))
            all_columns_file.append("pkts_transmitted_{}_{}".format(src, dest))
            all_columns_file.append("pkts_dropped_{}_{}".format(src, dest))

        for i in range(self.num_nodes ** 2):
            src = int(i / self.num_nodes)
            dest = int(i % self.num_nodes)
            all_columns_file.append("delay_e2e_{}_{}".format(src, dest))
            all_columns_file.append("delay_p10_{}_{}".format(src, dest))
            all_columns_file.append("delay_p20_{}_{}".format(src, dest))
            all_columns_file.append("delay_p50_{}_{}".format(src, dest))
            all_columns_file.append("delay_p80_{}_{}".format(src, dest))
            all_columns_file.append("delay_p90_{}_{}".format(src, dest))
            all_columns_file.append("jitter_{}_{}".format(src, dest))

        return all_columns_file

    '''
    Override
    '''
    def aggregateDataframes(self):
        df_train, df_test = pd.DataFrame(), pd.DataFrame()
        if not isdir(self.folder_cache) or not isfile(self.path_train) or not isfile(self.path_test) or not isfile(self.path_train_X) or not isfile(self.path_train_y) or not isfile(self.path_test_X) or not isfile(self.path_test_y) or not self.generate_tensors:
            files_exist = self.read_from_cache and isdir(self.folder_cache) and isfile(self.path_train) and isfile(self.path_test)
            if files_exist:
                df_train, df_test = self.init_dataframes_from_cache()
            else:
                df = pd.DataFrame()
                dirs = np.array([dir for dir in listdir(self.dir_datasets) if "results" in dir])
                for i, dir in enumerate(dirs):
                    df_current = self.getDataframeFromSimulation(join(*[self.dir_datasets, dir, self.simulation_filename]))
                    df = pd.concat([df, df_current], ignore_index=True)
                    print("INFO: Reading. Dataset = {}/{}.".format(i+1, len(dirs), end='\r', flush=True))

                df = df[self.input_columns + self.output_columns]
                df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 51)
                df_train.to_csv(path_or_buf=self.path_train, sep=' ',index=False, header=False)
                df_test.to_csv(path_or_buf=self.path_test, sep=' ',index=False, header=False)

        return (df_train, df_test)

    '''
    Override
    TODO test
    '''
    def get_dataframe_visualization(self):
        all_columns_file = self.get_all_columns_file()
        df_test_visualization = pd.DataFrame()
        intensities = np.array([])

        dirs = [dir for dir in listdir(self.dir_datasets) if "results" in dir and "Routing_SP_k_89" in dir]
        for f in dirs:
            path = join(*[self.dir_datasets, f, self.simulation_filename])
            intensity = int(f.split("/")[-1].split("_")[2])
            print("intensity", intensity)
            if isfile(path):
                df_current = pd.read_csv(path, sep=",", header=None, names=all_columns_file,index_col=False)
                df_current = df_current[self.input_columns + self.output_columns]
                df_current_samples = df_current.sample(frac=1, replace=False, random_state=1)
                df_test_visualization = pd.concat([df_test_visualization, df_current_samples], ignore_index=True)
                intensities = np.append(intensities, np.repeat(intensity, df_current_samples.shape[0]))
            else:
                print("no {}".format(path))
        df_test_visualization["intensity"] = intensities
        return df_test_visualization
