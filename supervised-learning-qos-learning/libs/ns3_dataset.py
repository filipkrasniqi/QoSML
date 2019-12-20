from columns import *

from os import listdir, scandir
from os.path import isfile, join,expanduser, isdir
import pandas as pd
import numpy as np
pd.set_option('precision', 9)

import torch
torch.set_printoptions(precision=9)

import itertools
from dataset_container import DatasetContainer

from enum import IntEnum

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
'''
Enum to define how to build train and test.
Set of dataset is composed of E environments, I intensities, S simulations.
For each e in E, you have different set of links distribution, i.e., capacities
For each i in I, you have different rates, given same e in E
For each s in S, fixed (i,e), you have same links, different rates but belonging to same distributions
SINGLE_E_UNSEEN_S: forces E = 1 and selects, given the combination of (S, I), a set of unseen S for each I
MULTIPLE_E_UNSEEN_S: same as above but without forcing E = 1. TODO to test
MULTIPLE_E_UNSEEN_E: for both train and test you have all the (s, i), but unseen e
MULTIPLE_E_UNSEEN_ES: combination between the last two. TODO to test
'''
class Scenario(IntEnum):
    LEVEL_1 = 0
    LEVEL_2 = 1
    LEVEL_3 = 2

'''
Extension of DatasetContainer for simulations obtained from ns3.
Split into test and train is made according to the scenario as explained above.

identifier: directory containing the execution of the simulation. Identifies the launch of a set of ns3 simulations
only_low: True if you want to consider only intensity = 0 in all your datasets
num_values_drop: by default, capping is not applied, but N periods are dropped (related to the beginning of the execution).
    N = num_values_drop
window_size: window size when extracting features s.t. mean, std, quantiles
test_less_intensities: if True, test is made only of intensity_{0, 4, 9}, otherwise of all the intensities
scenario: considered scenario (see above explanation)
train_ratio: % of simulations related to training. 1-train_ratio are those related to test
'''
class NS3Dataset(DatasetContainer, InMemoryDataset):
    def __init__(self,
                identifier = "simulation_v1",
                only_low = False,
                num_values_drop = 100,
                window_size = 51,
                quantiles_list = ["01"] + ["{}".format(i*10) for i in range(1, 10)] + ["99"],
                test_less_intensities = False,
                scenario = Scenario.LEVEL_1,
                train_ratio = 0.8,
                extract_also_quantiles = False,
                also_pyg = False,
                also_std_delay = False,
                **kwargs):
        'Initialize ns3 dataset'
        self.identifier = identifier
        self.only_low = only_low
        self.num_values_drop = num_values_drop
        self.window_size = window_size
        self.quantiles_list = quantiles_list
        self.test_less_intensities = test_less_intensities
        self.scenario = scenario
        self.datasets_output = 'datasets'
        self.train_ratio = train_ratio
        self.extract_also_quantiles = extract_also_quantiles
        self.also_pyg = also_pyg
        self.also_std_delay = also_std_delay

        root = join(*[expanduser('~'), 'notebooks', 'datasets', 'ns3'])
        transform = None
        pre_transform = None

        kwargs["root"], kwargs["transform"], kwargs["pre_transform"] = root, transform, pre_transform

        DatasetContainer.__init__(self, **kwargs)
        self.size_dataset_single_simulation = self.num_periods - self.window_size - self.num_values_drop + 1

        if self.also_pyg:
            InMemoryDataset.__init__(self, root, transform, pre_transform)
            # here check and if not present generate random numbers
            if self.only_test:
                self.data, self.slices = torch.load(self.processed_paths[1])
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])

            del self.dfs_e2e_train, self.dfs_e2e_test, self.X_train, self.X_test, self.y_train, self.y_test

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.processed_train_validate, self.processed_test]

    def download(self):
        pass

    def datalist_from_df(self, df_all, description="train and validate"):
        data_list = []  # list of (x, y, edge_idxs)
        for i, row in df_all.iterrows():
            # for each row, we define x, y, edge_index values.
            # Fixed node i, it has N values for X, i.e., the TM  having as origin himself, the delay_e2e with same consideration
            # each row is a row in my dataframe
            x = []
            y = []
            capacity = []
            # node attributes: traffic
            for j in range(self.max_num_nodes):
                x_n = [getattr(row, col) for col in df_all.columns if "mean_traffic_{}_".format(j) in col]
                y_n = [getattr(row, col) for col in df_all.columns if "mean_delay_e2e_{}_".format(j) in col]
                x.append(x_n)
                y.append(y_n)

            x = torch.Tensor(x)
            y = torch.Tensor(y)

            source_nodes, target_nodes = self.coo_from_adj()

            # edge attributes: capacity, dropped
            # important: need to stick to the edge_index order for the edges. Everytime I define edges property I need to loop with this behaviour or to follow (source, target) coo pair
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            for j in range(self.max_num_nodes ** 2):
                s = int(j / self.max_num_nodes)
                t = j % self.max_num_nodes
                if getattr(row, "capacity_{}_{}".format(s, t)) > 0:
                    c_n = getattr(row, "capacity_{}_{}".format(s, t))
                    capacity.append(c_n)
                if getattr(row, "capacity_{}_{}".format(t, s)) > 0:
                    c_n = getattr(row, "capacity_{}_{}".format(t, s))
                    capacity.append(c_n)
            capacity = torch.tensor(capacity)
            data = Data(x=x, edge_index=edge_index, edge_attr=capacity, y=y)
            data_list.append(data)

            if i > 0 and i % 1000 == 0:
                print("Processing {}: completed {} out of {}".format(description, i, df_all.shape[0]))
        return data_list

    def datalist_from_tensors(self, test):
        if test:
            description = "Test"
        else:
            description = "Train and validate"
        data_list = []  # list of (x, y, edge_idxs)
        num_rows = self.y_train.size(0)
        if test:
            num_rows = self.y_test.size(0)

        num_edge_features = 1   # eventually, only capacity# important: need to stick to the edge_index order for the edges. Everytime I define edges property I need to loop with this behaviour or to follow (source, target) coo pair
        source_nodes, target_nodes = self.coo_from_adj()
        edge_index = torch.cat([torch.tensor([source_nodes, target_nodes], dtype=torch.long), torch.tensor([target_nodes, source_nodes], dtype=torch.long)], 1)# torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        if self.scenario == Scenario.LEVEL_1 or self.X_test.size(1) % self.max_num_nodes == 0:
            num_edge_features = 0
            print("WARNING: no edge features")
        num_features = int((self.X_test.size(1) - num_edge_features * len(source_nodes)) / (self.max_num_nodes * (self.max_num_nodes - 1)))
        num_instances_per_feature = self.max_num_nodes * (self.max_num_nodes - 1)
        idxs_permutation = np.array([])
        for idx_node in range(self.max_num_nodes):
            idxs_permutation = np.append(idxs_permutation, [list(range(idx_node * (self.max_num_nodes - 1) + idx_feature * num_instances_per_feature, idx_node * (self.max_num_nodes - 1) + idx_feature * num_instances_per_feature + self.max_num_nodes - 1)) for idx_feature in range(num_features)])
        idxs_permutation = list(idxs_permutation.astype(int))
        for idx_tensor in range(num_rows):
            if test:
                idx_end_node_features = (self.max_num_nodes - 1) * self.max_num_nodes * num_features
                row_x_nodes, row_x_edge, row_y = self.X_test[idx_tensor][:idx_end_node_features], self.X_test[idx_tensor][idx_end_node_features:], self.y_test[idx_tensor]
            else:
                idx_end_node_features = (self.max_num_nodes - 1) * self.max_num_nodes * num_features
                row_x_nodes, row_x_edge, row_y = self.X_train[idx_tensor][:idx_end_node_features], self.X_train[idx_tensor][idx_end_node_features:],self.y_train[idx_tensor]
                # row_x, row_y = self.X_train[idx_tensor], self.y_train[idx_tensor]
            edge_attr = torch.cat([row_x_edge, row_x_edge], 0)
            # for each row, we define x, y, edge_index values.
            # Fixed node i, it has N values for X, i.e., the TM  having as origin himself, the delay_e2e with same consideration
            # each row is a row in my dataframe

            # node attributes: traffic
            row_x_nodes = row_x_nodes[idxs_permutation] # sort in order to provide the features in s.w.t. (mu_idx_node_<all_others>, std_idx_node_<all_others>)
            x = torch.Tensor(row_x_nodes.reshape(self.max_num_nodes, -1))
            y = torch.Tensor(row_y.reshape(self.max_num_nodes, -1))

            # edge attributes: capacity
            if self.scenario == Scenario.LEVEL_1:
                data = Data(x=x, edge_index=edge_index, y=y)
            else:
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

            if idx_tensor > 0 and idx_tensor % 1000 == 0:
                print("Processing {}: completed {} out of {}".format(description, idx_tensor, num_rows))
        return data_list

    def process(self):
        if self.also_pyg:
            print("Train and validate: {}".format(self.dfs_e2e_train.shape))
            data_list = self.datalist_from_tensors(test=False)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

            print("Test: {}".format(self.dfs_e2e_test.shape))
            data_list = self.datalist_from_tensors(test=True)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[1])

    '''
    Override
    '''
    def init_base_directory(self):
        self.dir_datasets = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', 'datasets', 'ns3'])
        self.dir_datasets_env = join(*[self.dir_datasets, self.topology, self.identifier])+"/"

    '''
    Override
    '''
    def init_cachefiles(self):
        if self.scenario.name not in self.cache_dir:
            postfix = ""
            if self.only_low:
                postfix = "_only_low"
            postfix += "_{}".format(self.scenario.name)
            self.cache_dir = '{}{}'.format(self.cache_dir, postfix)
        print("INFO: cache dir is {}".format(self.cache_dir))
        self.folder_cache = join(*[self.dir_datasets, self.topology, self.identifier, self.cache_dir])
        super().init_cachefiles()

    '''
    Override
    '''
    def init_columns(self):
        super().init_columns()

        filter_columns = ["mean", "std"]
        prefixes_cols_self_edges = ["{}_{}".format(i, i) for i in range(self.max_num_nodes)]
        prefixes_cols_edges = self.get_edge_prefixes()

        if self.extract_also_quantiles:
            self.input_columns = self.feature_cols("mean", remove_self_loops=True) + self.feature_cols("std", remove_self_loops=True) + self.quantiles_cols()
        else:
            self.input_columns = self.feature_cols("mean", remove_self_loops=True) + self.feature_cols("std", remove_self_loops=True)

        self.input_columns += self.capacity_cols
        filter_columns += ["capacity"]
        self.input_columns = [col for col in self.input_columns if any(column_for_tensor in col for column_for_tensor in filter_columns) and (("capacity" not in col and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges)) or ("capacity" in col and any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_edges)))]

        self.output_columns = self.feature_cols("mean", build_columns_only_delay_e2e, remove_self_loops=True)
        if self.also_std_delay:
            self.output_columns += self.feature_cols("std", build_columns_only_delay_e2e, remove_self_loops=True)
        self.output_columns.append("intensity")
        self.output_columns.append("simulation")

        # filter by name in the column, for OD flow related features (aka, all columns that are not capacity) -> remove autoreferred nodes, for link related features: remove those that are not edges
        self.tensor_input_columns = [col for col in self.input_columns if any(column_for_tensor in col for column_for_tensor in filter_columns) and (("capacity" not in col and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges)) or ("capacity" in col and self.scenario != Scenario.LEVEL_1 and any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_edges)))]
        self.tensor_output_columns = [col for col in self.output_columns if "delay_e2e_" in col]

    '''
    When extracting feature from another (eg: mean from window of traffic),
    I create a column whose name is <feature_to_ext>_<feature_from>_<O>_<D>
    '''
    def feature_cols(self, feature, fun_columns = build_columns_only_traffic, remove_self_loops = True):
        columns = fun_columns(self.num_nodes)
        # remove self inner loops
        if remove_self_loops:
            columns = [col for i, col in enumerate(columns) if int(i / self.num_nodes) != i % self.num_nodes]
        return ["{}_{}".format(feature, col) for col in columns]

    '''
    Calls feature_cols to build the columns related to quantiles
    '''
    def quantiles_cols(self, quantile = None):
        if quantile is None:
            cols = []
            for q in self.quantiles_list:
                cols += self.feature_cols("q{}".format(q))
            return cols
        else:
            return self.feature_cols("q{}".format(quantile))

    '''
    Override
    '''
    def get_info_simulation(self):
        filename_sim = join(*[self.dir_datasets_env, "simulation.txt"])
        sim_f = open(filename_sim , 'r' ) # content: (X, X, num_periods, num_nodes)
        for count_line, line in enumerate(sim_f.readlines()):
            if(count_line == 0):
                _1,_2,num_periods,num_nodes =  line.split(" ")
                num_periods, num_nodes = int(num_periods), int(num_nodes)

        sim_f.close()
        return num_periods, num_nodes

    def intensity_dir(self, intensity, simulation):
        return "intensity_{}_{}".format(intensity, simulation)

    def environment_dir(self, capacity, pd):
        return "environment_{}_{}".format(capacity, pd)

    '''
    Override
    '''
    def aggregateDataframes(self):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        if not isdir(self.folder_cache) or not isfile(self.path_train) or not isfile(self.path_test) or not isfile(self.path_train_X) or not isfile(self.path_train_y) or not isfile(self.path_test_X) or not isfile(self.path_test_y) or not self.generate_tensors:
            files_exist = self.read_from_cache and isdir(self.folder_cache) and isfile(self.path_train) and isfile(self.path_test)
            if files_exist:
                df_train, df_test = self.init_dataframes_from_cache(is_numpy=False)
            else:
                print("INFO: starting caching unified datasets")
                count_not_working = 0
                log_output = "\nList of not existing:\n"
                list_dfs_train, list_dfs_test = [], []
                for idx, (simulation, intensity, capacity, prop_delay) in enumerate(self.possible_existing_combinations):
                    # check if it is inside the array of combinations
                    if (simulation, intensity, capacity, prop_delay) in self.combinations_SICP:
                        e2e_dfs = self.getDataframeFromSimulation(self.intensity_dir(intensity, simulation), "environment_{}_{}".format(capacity, prop_delay))
                        df = pd.concat(e2e_dfs, axis=1, sort=False, join='inner')
                        if capacity in self.test_capacities and prop_delay in self.test_pdelays and intensity in self.test_distinct_intensities and simulation in self.test_simulations:
                            # df_test = pd.concat([df_test, pd.concat(e2e_dfs, axis=1, sort=False,join='inner')])
                            list_dfs_test.append(df)
                        else:
                            # df_train = pd.concat([df_train, pd.concat(e2e_dfs, axis=1, sort=False,join='inner')])
                            list_dfs_train.append(df)
                    else:
                        print("WARNING: ({},{}, {}, {}) not ready yet!".format(simulation, intensity, capacity, prop_delay))
                        log_output += "S = {}, I = {}, C = {}, P = {}\n".format(simulation, intensity, capacity, prop_delay)
                        count_not_working += 1

                    print("INFO: Reading. S = {}/{}, I = {}/{}, C = {}/{}, P = {}/{}.".format(simulation+1, len(self.distinct_simulations), intensity+1, len(self.distinct_intensities), capacity+1, len(self.distinct_capacities), prop_delay + 1, len(self.distinct_pdelays)), end='\r', flush=True)

                df_train = pd.concat(list_dfs_train)
                del list_dfs_train
                df_test = pd.concat(list_dfs_test)
                del list_dfs_test
                # now normalize and cap
                print("INFO: Starting normalization...")
                continuous_columns = [ col for i, col in enumerate(self.input_columns) if "traffic" in col and i%self.max_num_nodes != int(i/self.max_num_nodes)]
                if self.do_normalization:
                    df_train = self.getNormalizedDataframe(df_train, continuous_columns)
                    df_test = self.getNormalizedDataframe(df_test, continuous_columns)
                else:
                    print("WARNING: Skipping normalization...")

                to_cap_columns = [ col for i, col in enumerate(self.input_columns) if "traffic" in col and i%self.max_num_nodes != int(i/self.max_num_nodes)]
                to_cap_columns += [ col for i, col in enumerate(self.output_columns) if "delay" in col and i%self.max_num_nodes != int(i/self.max_num_nodes)]

                print("INFO: Starting capping...")
                if self.do_capping:
                    df_train = self.getCappedDataframe(df_train, to_cap_columns)
                    df_test = self.getCappedDataframe(df_test, to_cap_columns)
                else:
                    print("WARNING: Skipping capping...")
                # check if it is consistent (can not be in case of not enough data)
                if df_train.shape[0] <= 0 or df_test.shape[0] <= 0:
                    raise ValueError("ERROR: not enough data. Dataset is empty! Shapes of train and test are, respectively, {} and {}".format(df_train.shape, df_test.shape))
                # cache to file
                df_train.to_csv(path_or_buf=self.path_train, sep=' ',index=False, header=False)
                df_test.to_csv(path_or_buf=self.path_test, sep=' ',index=False, header=False)

                print("SUCCESS: cached train and test, respectively of shapes {}, {}".format(df_train.shape, df_test.shape))
        else:
            print("WARNING: skipping reading dataframes.")

        return df_train, df_test

    def get_df(self, filename, cols_fun, intensity = "intensity_0_0", simulation = "environment_0_0"):
        folder_dataset = "{}/{}/{}/{}/{}/{}/".format(self.dir_datasets, self.topology, self.identifier, intensity, simulation, self.datasets_output)
        return pd.read_csv(folder_dataset+self.prefix_filename+filename, sep=" ", header=None, names=cols_fun(self.num_nodes),index_col=False)

    '''
    Returns single row dataframe related to routing
    '''
    def get_routing_df(self, intensity = "intensity_0_0", simulation = "environment_0_0"):
        return self.get_df("routing.txt", build_columns_routing)

    '''
    Returns single row dataframe related to capacities
    '''
    def get_capacities_df(self):
        return self.get_df("links.txt", build_columns_only_links).loc[:, build_columns_capacity(self.num_nodes)]

    '''
    Override.
    Focus on how to distribute E, I, S for each scenario.
    Filename: intensity_<intensity>_<simulation>/environment_<capacity>_<delay>
    '''
    def init_variables_dataset(self):
        self.prefix_filename = "{}_{}_1_{}_P_".format(
            self.num_nodes,
            self.num_nodes,
            self.num_periods
        )
        # read intensities from directory. Read current env and look for dirs starting with "intensity"
        possible_intensities = [f.path for f in scandir(self.dir_datasets_env) if f.is_dir() and "intensity" in f.path]
        if len(possible_intensities) > 0:
            self.intensities = [intensity.split("/")[-1] for intensity in possible_intensities]
        else:
            raise ValueError("ERROR: no directories with intensity")
        self.distinct_intensities = []
        self.distinct_simulations = []
        for intensity in self.intensities:
            only_intensity = int(intensity.split("_")[1])
            only_simulation = int(intensity.split("_")[2])
            if only_intensity not in self.distinct_intensities:
                self.distinct_intensities.append(only_intensity)
            if only_simulation not in self.distinct_simulations:
                self.distinct_simulations.append(only_simulation)
        self.distinct_intensities = sorted(self.distinct_intensities)
        self.distinct_simulations = sorted(self.distinct_simulations)
        self.intensities = sorted(self.intensities, key = lambda x: int(x.split("_")[1]) * len(self.distinct_simulations) + int(x.split("_")[2]))

        if self.only_low:
            self.intensities = [self.intensities[0]]

        self.low_intensity = self.intensities[0]
        if not self.only_low:
            self.high_intensity = self.intensities[-1]
            self.medium_intensity = self.intensities[int((len(self.intensities)-1) / 2)]

        all_prop_delays = [int(f.path.split("/")[-1].split("_")[-2]) for f in scandir(join(*[self.dir_datasets_env, self.intensities[-1]])) if f.is_dir() and "environment_" in f.path]
        self.num_prop_delays = max(all_prop_delays)

        self.distinct_pdelays = sorted(list(set([int(f.path.split("/")[-1].split("_")[-1]) for f in scandir(join(*[self.dir_datasets_env, self.intensities[-1]])) if f.is_dir() and "environment_" in f.path])))
        self.distinct_capacities = sorted(list(set([int(f.path.split("/")[-1].split("_")[-2]) for f in scandir(join(*[self.dir_datasets_env, self.intensities[-1]])) if f.is_dir() and "environment_" in f.path])))
        # reading all possible environments and ordering them by considering the number x s.t. simulation_<x>
        if self.scenario == Scenario.LEVEL_1:
            print("WARNING: considering only one distribution of (capacity, propagation delay)")
            self.distinct_pdelays = [self.distinct_pdelays[0]]
            self.distinct_capacities = [self.distinct_capacities[0]]
        elif self.scenario == Scenario.LEVEL_2:
            print("WARNING: considering only one distribution of propagation delay but multiple distinct_capacities")
            self.distinct_pdelays = [self.distinct_pdelays[0]]
        # unaccepted case: MULTIPLE_E_UNSEEN_S and single S
        if self.scenario == Scenario.LEVEL_1 and len(self.distinct_simulations) <= 10:
            raise ValueError("ERROR: trying to generalize on unseen simulations with low amount of simulations")
        self.combinations_SICP = []
        print("INFO: checking combinations (I, S, E) having the dataset.")
        self.possible_existing_combinations = list(itertools.product(self.distinct_simulations, self.distinct_intensities, self.distinct_capacities, self.distinct_pdelays))
        self.max_simulation = 0     # represents simulation having at least one dataset
        self.max_capacity = 0       # represents environment having at least one dataset
        self.max_intensity = 0      # represents intensity having at least one dataset
        self.max_pdelay = 0         # represents pdelay having at least one dataset
        for simulation in self.distinct_simulations:
            for intensity in self.distinct_intensities:
                for pdelay in self.distinct_pdelays:
                    for capacity in self.distinct_capacities:
                        dir_datasets_intensity = join(*[self.dir_datasets_env, self.intensity_dir(intensity, simulation)])
                        current_dataset = join(*[dir_datasets_intensity, "environment_{}_{}".format(capacity, pdelay), self.datasets_output])
                        if isdir(current_dataset) and len(listdir(current_dataset)) > 0:
                            self.combinations_SICP.append((simulation, intensity, capacity, pdelay))
                            if capacity > self.max_capacity:
                                self.max_capacity = capacity
                            if simulation > self.max_simulation:
                                self.max_simulation = simulation
                            if intensity > self.max_intensity:
                                self.max_intensity = intensity
                            if pdelay > self.max_pdelay:
                                self.max_pdelay = pdelay

        if self.max_simulation < len(self.distinct_simulations) - 1:
            print("WARNING: not all simulations are ready")

        if self.max_capacity < len(self.distinct_capacities) - 1:
            print("WARNING: not all environments are ready")

        if self.max_intensity < len(self.distinct_intensities) - 1:
            print("WARNING: not all intensities are ready")

        if self.max_pdelay < len(self.distinct_pdelays) - 1:
            print("WARNING: not all prop delays are ready")

        print("INFO LOG: # (I, S, E) = {}. Ordered by (intensity, simulation).".format(len(self.combinations_SICP)))

        if self.test_less_intensities:
            self.test_distinct_intensities = [0, int(((len(self.distinct_intensities) - 1) / 2)), len(self.distinct_intensities) - 1] # low, medium and high intensities
            print("WARNING: test is not made of all the intensities, only of {}".format(self.test_distinct_intensities))
        else:
            self.test_distinct_intensities = self.distinct_intensities # as a default case, I consider when testing all the intensities

        if self.scenario == Scenario.LEVEL_1:
            self.train_simulations = self.distinct_simulations[0:int(self.train_ratio * len(self.distinct_simulations))]
            self.test_simulations = self.distinct_simulations[len(self.train_simulations):]

            self.train_capacities = self.test_capacities = [self.distinct_capacities[0]]
            self.train_pdelays = self.test_pdelays = [self.distinct_pdelays[0]]

        elif self.scenario == Scenario.LEVEL_2:
            self.train_simulations = self.test_simulations = self.distinct_simulations

            self.train_capacities = self.distinct_capacities[0:int(self.train_ratio * len(self.distinct_capacities))]
            self.test_capacities = self.distinct_capacities[len(self.train_capacities):]
            self.train_pdelays = self.test_pdelays = [self.distinct_pdelays[0]]

        elif self.scenario == Scenario.LEVEL_3:
            self.train_simulations = self.test_simulations = self.distinct_simulations
            # here ratios can't be defined as the same. I take capacity = 50%, PDs = 40% -> 0.5 * 0.4 = 20% = test size

            self.train_capacities = self.distinct_capacities[0:int(0.5 * len(self.distinct_capacities))]
            self.test_capacities = self.distinct_capacities[len(self.train_capacities):]

            self.train_pdelays = self.distinct_pdelays[0:int(0.6 * len(self.distinct_pdelays))]
            self.test_pdelays = self.distinct_pdelays[len(self.train_pdelays):]

    '''
    Override.
    Given an intensity (already considering both i and s) and an environment, it returns the associated dataframe read from file
    '''
    def getDataframeFromSimulation(self, intensity, sim, window_size = None):
        if window_size is None:
            window_size = self.window_size
        folder_dataset = "{}/{}/{}/{}/{}/{}/".format(self.dir_datasets, self.topology, self.identifier, intensity, sim, self.datasets_output)
        filename_e2e = folder_dataset+self.prefix_filename+"delaye2e.txt"
        filename_TM = folder_dataset+self.prefix_filename+'TM.txt'
        filename_links = folder_dataset+self.prefix_filename+'links.txt'
        filename_dropped = folder_dataset+self.prefix_filename+'dropped.txt'

        # reading values for e2e and computing delay per packet
        df_delay_e2e = pd.read_csv(filename_e2e, sep=" ", header=None, names=self.delay_cols,index_col=False).drop(list(range(self.num_values_drop))+[self.num_periods], axis=0)
        # remove self loops
        filtered_delay_cols = [col for i, col in enumerate(df_delay_e2e.columns) if int(i / self.num_nodes) != i % self.num_nodes]
        df_delay_e2e = df_delay_e2e[filtered_delay_cols]
        df_delay_e2e.reset_index(inplace=True)

        df_delay_e2e = self.imputing(df_delay_e2e)
        df_delay_e2e["period"] = pd.to_timedelta(df_delay_e2e.index.values * 100, unit="ms")
        df_delay_e2e.set_index('period', inplace=True)

        df_e2e = df_delay_e2e[filtered_delay_cols].rolling(window_size).mean().dropna()
        df_e2e.columns = self.feature_cols("mean", build_columns_only_delay_e2e, remove_self_loops=True)

        df_temp = pd.DataFrame()
        if self.also_std_delay:
            df_temp = df_delay_e2e[filtered_delay_cols].rolling(window_size).std().dropna()
            df_temp.columns = self.feature_cols("std", build_columns_only_delay_e2e, remove_self_loops=True)

        df_e2e = pd.concat([df_e2e, df_temp], axis=1, sort=False)

        # computing dataframe for timeseries
        df_TM_total = pd.read_csv(filename_TM, sep=" ", header=None, names = self.traffic_cols,index_col=False)

        if self.impute_traffic:
            df_TM_total = self.imputing(df_TM_total, "traffic")

        # remove self loops
        filtered_traffic_cols = [col for i, col in enumerate(df_TM_total.columns) if int(i / self.num_nodes) != i % self.num_nodes]
        df_TM_total = df_TM_total[filtered_traffic_cols]

        df_TM_base_ts = df_TM_total.drop(list(range(self.num_values_drop))+[self.num_periods], axis=0)
        df_TM_base_ts.reset_index(inplace=True)

        df_TM_base_ts["period"] = pd.to_timedelta(df_TM_base_ts.index.values * 100, unit="ms")
        df_TM_base_ts.set_index('period', inplace=True)

        df_ts = df_TM_base_ts[filtered_traffic_cols].rolling(window_size).mean().dropna()
        df_ts.columns = self.feature_cols("mean", remove_self_loops=True)# ["mean_{}".format(col) for col in filtered_traffic_cols]

        df_temp = df_TM_base_ts[filtered_traffic_cols].rolling(window_size).std().dropna()
        df_temp.columns = self.feature_cols("std", remove_self_loops=True)# ["std_{}".format(col) for col in filtered_traffic_cols]

        df_ts = pd.concat([df_ts, df_temp], axis=1, sort=False)

        if self.extract_also_quantiles:
            for quantile in self.quantiles_list:
                quantile = float("0.{}".format(quantile))
                df_ts =  pd.concat([df_ts, df_TM_base_ts[filtered_traffic_cols].rolling(window_size).quantile(quantile, interpolation='midpoint').dropna()], axis=1, sort=False)
                mapping = {t_col: q_col for t_col, q_col in zip(filtered_traffic_cols, self.quantiles_cols(quantile))}
                df_ts.rename(columns=mapping, inplace = True)

        # capacity
        columns_capacities = build_columns_capacity(self.num_nodes)
        df_link_single = pd.read_csv(filename_links, sep=" ", header=None, names=build_columns_only_links(self.num_nodes),index_col=False)
        df_capacities_single = df_link_single[columns_capacities]
        # get columns from this line
        filtered_capacity_cols = [col for col in df_capacities_single.columns if
                                  df_capacities_single[col].values[0] > 0]
        df_capacities_single = df_capacities_single[filtered_capacity_cols]
        df_capacities = df_capacities_single.loc[np.repeat(df_capacities_single.index.values, df_ts.shape[0])]

        #just add to this, then I will merge
        df_e2e["intensity"] = intensity
        df_e2e["simulation"] = sim

        cols = [col for col in df_e2e.columns if "delay" in col]
        df_e2e.loc[:, cols] = df_e2e.loc[:, cols].multiply(10**-9) # nanoseconds/packet -> seconds/packet

        # change according to current implementation of input and targets. Now it is e2e = f(TM, D) being D = cumulative drops.
        target_dfs = [df_e2e]
        input_dfs = [df_ts, df_capacities]

        for df in input_dfs:
            df.reset_index(drop=True, inplace=True)
        for df in target_dfs:
            df.reset_index(drop=True, inplace=True)

        df_target = pd.concat(target_dfs, axis=1, sort=False)
        df_input = pd.concat(input_dfs, axis=1, sort=False)

        return df_input, df_target

    def getDataframeLoadFromSimulation(self, intensity, sim, window_size = None):
        if window_size is None:
            window_size = self.window_size
        folder_dataset = "{}/{}/{}/{}/{}/{}/".format(self.dir_datasets, self.topology, self.identifier, intensity, sim, self.datasets_output)
        filename_load = folder_dataset+self.prefix_filename+'load.txt'
        filename_dropped = folder_dataset+self.prefix_filename+'dropped.txt'

        # computing dataframe for timeseries
        df_load_total = pd.read_csv(filename_load, sep=" ", header=None, names = self.load_cols,index_col=False)

        df_load_base_ts = df_load_total.drop(list(range(self.num_values_drop))+[self.num_periods], axis=0)
        df_load_base_ts.reset_index(inplace=True)

        df_load_base_ts["period"] = pd.to_timedelta(df_load_base_ts.index.values * 100, unit="ms")
        df_load_base_ts.set_index('period', inplace=True)

        df_load = df_load_base_ts[self.load_cols].rolling(window_size).mean().dropna()
        df_load.columns = ["mean_{}".format(col) for col in self.load_cols]

        #just add to this, then I will merge
        df_load["intensity"] = intensity
        df_load["simulation"] = sim

        df_load.reset_index(drop=True, inplace=True)

        return df_load

    def getDataframeDroppedFromSimulation(self, intensity, sim, window_size = None):
        if window_size is None:
            window_size = self.window_size
        folder_dataset = "{}/{}/{}/{}/{}/{}/".format(self.dir_datasets, self.topology, self.identifier, intensity, sim, self.datasets_output)
        filename_dropped = folder_dataset+self.prefix_filename+'dropped.txt'

        # computing dataframe for timeseries
        df_dropped_total = pd.read_csv(filename_dropped, sep=" ", header=None, names = self.dropped_cols,index_col=False)

        df_dropped_base_ts = df_dropped_total.drop(list(range(self.num_values_drop))+[self.num_periods], axis=0)
        df_dropped_base_ts.reset_index(inplace=True)

        df_dropped_base_ts["period"] = pd.to_timedelta(df_dropped_base_ts.index.values * 100, unit="ms")
        df_dropped_base_ts.set_index('period', inplace=True)

        df_dropped = df_dropped_base_ts[self.dropped_cols].rolling(window_size).mean().dropna()
        df_dropped.columns = ["mean_{}".format(col) for col in self.dropped_cols]

        #just add to this, then I will merge
        df_dropped["intensity"] = intensity
        df_dropped["simulation"] = sim

        df_dropped.reset_index(drop=True, inplace=True)

        return df_dropped

    '''
    Returns train/test unique values for simulation and intensities
    '''
    def get_unique_simulation_intensities(self, test = True):
        if test:
            df_current = self.dfs_e2e_test
        else:
            df_current = self.dfs_e2e_train
        return df_current.simulation.unique(), df_current.intensity.unique()

    '''
    Override
    The dataset is complete, what I will do is to get traffic, delay columns, and assign the intensity
    '''
    def get_dataframe_visualization(self, test_environments, test_int_sim, window_size = None):
        df_wdw_visualization = pd.DataFrame()
        df_visualization = pd.DataFrame()
        df_visualization_dropped = pd.DataFrame()
        intensities_df_tm, environments_df_tm, intensities, environments, intensities_dropped, environments_dropped = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        if window_size is None:
            window_size = self.window_size

        test_f = list(itertools.product(test_int_sim, test_environments))
        cur = 0

        for i, (intensity_simulation, environment) in enumerate(test_f):
            # get raw data for TM and delay
            prefix = "{}{}/{}/{}/{}".format(self.dir_datasets_env, str(intensity_simulation), str(environment), self.datasets_output, self.prefix_filename)
            path_delay = prefix+"delaye2e.txt"
            path_tm = prefix+"TM.txt"
            path_links = prefix+"links.txt"

            if isfile(path_tm) and isfile(path_delay):
                # get windowed dataframe
                df_input, df_target = self.getDataframeFromSimulation(intensity_simulation, environment, window_size)
                df_load = self.getDataframeLoadFromSimulation(intensity_simulation, environment, window_size).drop(['intensity', 'simulation'], axis=1)
                # df_dropped = self.getDataframeDroppedFromSimulation(intensity_simulation, environment, window_size).drop(['intensity', 'simulation'], axis=1)
                folder_dataset = "{}/{}/{}/{}/{}/{}/".format(self.dir_datasets, self.topology, self.identifier, intensity_simulation, environment, self.datasets_output)
                filename_dropped = folder_dataset+self.prefix_filename+'dropped.txt'

                # computing dataframe for timeseries
                df_dropped = pd.read_csv(filename_dropped, sep=" ", header=None, names = self.dropped_cols,index_col=False).drop(list(range(self.num_values_drop))+[self.num_periods], axis=0)
                intensities_dropped = np.append(intensities_dropped, np.repeat(intensity_simulation, df_dropped.shape[0]))
                environments_dropped = np.append(environments_dropped, np.repeat(environment, df_dropped.shape[0]))
                df_visualization_dropped = pd.concat([df_visualization_dropped, df_dropped], ignore_index=True)

                df_it = pd.concat([df_input, df_load, df_target], axis=1, sort=False,join='inner').reindex()
                df_wdw_visualization = pd.concat([df_wdw_visualization, df_it], ignore_index=True)
                intensities = np.append(intensities, np.repeat(intensity_simulation, df_it.shape[0]))
                environments = np.append(environments, np.repeat(environment, df_it.shape[0]))

                current_df_delay = pd.read_csv(path_delay, sep=" ", header=None, names=self.delay_cols,index_col=False).drop(list(range(self.num_values_drop))+[self.num_periods], axis=0).reset_index(drop=True)

                current_df_delay = current_df_delay.multiply(10**-9) # nanoseconds/packet -> seconds/packet

                current_df_tm = pd.read_csv(path_tm, sep=" ", header=None, names=self.traffic_cols,index_col=False).drop(list(range(self.num_values_drop))+[self.num_periods], axis=0).reset_index(drop=True)
                intensities_df_tm = np.append(intensities_df_tm, np.repeat(intensity_simulation, current_df_tm.shape[0]))
                environments_df_tm = np.append(environments_df_tm, np.repeat(environment, current_df_tm.shape[0]))

                columns_capacities = build_columns_capacity(self.num_nodes)
                df_link_single = pd.read_csv(path_links, sep=" ", header=None, names=build_columns_only_links(self.num_nodes),index_col=False)
                df_capacities_single = df_link_single[columns_capacities]
                df_capacities = df_capacities_single.loc[np.repeat(df_capacities_single.index.values, current_df_tm.shape[0])].reset_index(drop=True)

                df_t_d_c = pd.concat([current_df_tm, df_capacities, current_df_delay], axis=1, sort=False,join='inner').reindex()
                df_visualization = pd.concat([df_visualization, df_t_d_c], ignore_index=True)
            else:
                print("ERROR: no {} or {}".format(path_tm, path_delay))
            cur += 1
        df_wdw_visualization = df_wdw_visualization.reset_index()
        df_wdw_visualization["intensity_simulation"] = intensities
        df_wdw_visualization["environment"] = environments
        df_visualization["intensity_simulation"] = intensities_df_tm
        df_visualization["environment"] = environments_df_tm
        df_visualization_dropped["intensity_simulation"] = intensities_dropped
        df_visualization_dropped["environment"] = environments_dropped

        return df_wdw_visualization, df_visualization, df_visualization_dropped

    def get_capacities(self, idx_links):
        example_intensity, example_simulation, example_pd = self.distinct_intensities[0], self.distinct_simulations[0], self.distinct_pdelays[0]
        capacities = {}
        for capacity in self.distinct_capacities:
            filename_links = join(*[self.dir_datasets_env, self.intensity_dir(example_intensity, example_simulation),
                                  self.environment_dir(capacity, example_pd), "links.txt"])
            links_f = open(filename_links, 'r')
            for count_line, line in enumerate(links_f.readlines()):
                if count_line - 1 in idx_links:
                    links_props = line.split(" ")
                    link = "{}-{}".format(links_props[0], links_props[1])
                    if link not in capacities.keys():
                        capacities[link] = []
                    capacity_val_str = links_props[3]
                    if "Mbps" in capacity_val_str:
                        coeff, munit = 1, "Mbps"
                    else:
                        coeff, munit = 0.001, "Kbps"
                    capacities[link].append(int(int(capacity_val_str.split(munit)[0]) * coeff))
        return capacities

    # Returns map with keys: the links
    def get_pds(self, idx_links):
        example_intensity, example_simulation, example_capacity = self.distinct_intensities[0], self.distinct_simulations[0], self.distinct_capacities[0]
        pdelays = {}
        for pdelay in self.distinct_pdelays:
            filename_links = join(*[self.dir_datasets_env, self.intensity_dir(example_intensity, example_simulation),
                                  self.environment_dir(example_capacity, pdelay), "links.txt"])
            links_f = open(filename_links, 'r')
            for count_line, line in enumerate(links_f.readlines()):
                if count_line - 1 in idx_links:
                    links_props = line.split(" ")
                    link = "{}-{}".format(links_props[0], links_props[1])
                    if link not in pdelays.keys():
                        pdelays[link] = []
                    pdelay_val_str = links_props[2]
                    if "ms" in pdelay_val_str:
                        coeff, munit = 1, "ms"
                    else:
                        print("ERROR: {}".format(pdelay_val_str))
                    pdelays[link].append(int(int(pdelay_val_str.split(munit)[0]) * coeff))
        return pdelays

    # Union of all PDs
    def get_pds_union(self):
        example_intensity, example_simulation, example_capacity = self.distinct_intensities[0], self.distinct_simulations[0], self.distinct_capacities[0]
        pdelays = []
        for pdelay in self.distinct_pdelays:
            filename_links = join(*[self.dir_datasets_env, self.intensity_dir(example_intensity, example_simulation),
                                  self.environment_dir(example_capacity, pdelay), "links.txt"])
            links_f = open(filename_links, 'r')
            for count_line, line in enumerate(links_f.readlines()):
                links_props = line.split(" ")
                pdelay_val_str = links_props[2]
                if "ms" in pdelay_val_str:
                    coeff, munit = 1, "ms"
                else:
                    print("ERROR: {}".format(pdelay_val_str))
                pdelays.append(int(int(pdelay_val_str.split(munit)[0]) * coeff))
        return pdelays

    # Returns a dataset of same length of test completing the intensity, simulation, capacity, propagation delay part
    def get_test_iscp_dataframe(self):
        test_combinations = list(itertools.product(self.test_simulations, self.test_distinct_intensities, self.test_capacities, self.test_pdelays))
        dataset_length = self.getDataframeFromSimulation("intensity_0_0", "environment_0_0")[0].shape[0]
        simulations, intensities, capacities, prop_delays = np.array([]), np.array([]), np.array([]), np.array([])
        for idx, (simulation, intensity, capacity, prop_delay) in enumerate(test_combinations):
            simulations, intensities, capacities, prop_delays = np.append(simulations, np.repeat(int(simulation), dataset_length)), \
                                                                np.append(intensities, np.repeat(int(intensity), dataset_length)), \
                                                                np.append(capacities, np.repeat(int(capacity), dataset_length)), \
                                                                np.append(prop_delays, np.repeat(int(prop_delay), dataset_length))
        return pd.DataFrame(data={"Simulation": simulations, "Intensity": intensities, "Capacity": capacities, "P.Delay": prop_delays})
    '''
    Initializes values of train + validate and indices to have CV given a k value
    The idea is to split the dataset into train and validate, not with a simple
    random draw from dataset, but selecting ranges in such a way that
    validation follows the same constraint as test, i.e., it is related to
    unseen combination of (capacity, intensity)
    TODO test; what about running the same but with bootstrapping?
    '''
    def init_cv_v2(self, cv_k, range_columns = None, range_output_columns = None):
        print("INFO: starting CV split")

        if range_columns is None:
            range_columns = list(range(self.X_train.size(1)))
        if range_output_columns is None:
            range_output_columns = list(range(self.max_num_nodes ** 2))

        # X_train = self.filter_tensor_by_column(self.X_train, range_columns)
        # y_train = self.filter_tensor_by_column(self.y_train, range_output_columns)
        print("INFO: overall shape of train dataset: {}".format(self.X_train.shape))
        # consider train and validate together
        self.current_cv_k = cv_k

        print("INFO: final shape of train dataset: {}".format(self.X_train.shape))

        if self.subsets_approximate_fraction > 1:
            print("ERROR: subsets_approximate_fraction is a fraction, thus must be < 1")

        delta_rows_per_intensity = self.size_dataset_single_simulation * len(self.intensities)
        print("INFO: a single simulation contains {} values, each intensity contains {} values".format(self.size_dataset_single_simulation, delta_rows_per_intensity))
        '''
        Not relevant the order, but still, the entire dataframe (and consequently the tensors)
        are made in the following way:
        - fix intensity
            - fix simulation
                - fix capacity
                    - fix propagation delay
                        - obtain <size_dataset_single_simulation> values
        '''
        # TODO define depending on the scenario.
        if self.scenario == Scenario.LEVEL_1:
            raise ValueError("Validate in a case on which we change only the capacity, but consider all intensities")
        elif self.scenario == Scenario.LEVEL_2:
            raise ValueError("Validate in a case on which we change only the capacity, but consider all intensities")
        else:
            ratio_validation = 0.8
            num_single_datasets = int(self.X_train.size(0) / self.size_dataset_single_simulation)
            range_datasets = list(range(num_single_datasets))
            print(num_single_datasets)
            self.validation_indices = []
            # num_sims_fixed_simulation, num_sims_fixed_intensities, num_sims_fixed_capacities = 10 * 50 * 10, 50 * 10, 10
            for k in range(self.current_cv_k):
                # for each CV, sample indices of simulations, not of instances
                validation_idxs = np.random.choice(range_datasets, int(ratio_validation * num_single_datasets), replace=False)
                self.validation_indices.append(validation_idxs)

    def get_current_data_loaders_v2(self, current_k, batch_size, window_size = None):
        assert self.validation_indices is not None and self.X_train is not None and self.y_train is not None and current_k < self.current_cv_k, "Something is wrong"
        indices_validate = np.array([])
        for idx_start in self.validation_indices[current_k]:
            idxs = list(range(idx_start * self.size_dataset_single_simulation, (idx_start + 1) * self.size_dataset_single_simulation))
            indices_validate = np.append(indices_validate, idxs)
        indices_train = np.array(list(set(list(range(self.X_train.size(0)))) - set(indices_validate)))# np.array([idx for idx in range(self.X_train.size(0)) if idx not in indices_validate])
        indices_validate = np.random.choice(indices_validate, len(indices_validate), replace=False)
        indices_train = np.random.choice(indices_train, len(indices_train), replace=False)
        return self.get_data_loaders(batch_size, indices_train, indices_validate, window_size)

    def get_current_num_environments(self, test = False):
        if test:
            simulations, intensities, capacities, pdelays = self.test_simulations, [0, 4, 9], self.test_capacities, self.test_pdelays
        else:
            simulations, intensities, capacities, pdelays = self.train_simulations, self.intensities, self.train_capacities, self.train_pdelays

        return len(intensities) * len(capacities) * len(pdelays), len(capacities) * len(pdelays), len(pdelays)

    # TODO fix this: must become that allows to init a tensor from another dataframe providng also the name -> for PCA
    def getTensorsFromDataframesv2(self, filter_columns = ["mean", "capacity"]):
        if self.need_init_tensors():
            print("INFO: Reading tensor from cache...")
            X_train, y_train = (None, None)
            if not self.only_test:
                X_train = torch.load(self.path_train_X)
                y_train = torch.load(self.path_train_y)

            X_test = torch.load(self.path_test_X)
            y_test = torch.load(self.path_test_y)
        else:
            print("INFO: Computing tensors and writing to file...")
            df_train, df_test = self.dfs_e2e_train, self.dfs_e2e_test

            X_train = df_train.loc[:, self.tensor_input_columns]
            y_train = df_train.loc[:, self.tensor_output_columns]

            for col in y_train.columns:
                y_train.loc[:, col] = pd.to_numeric(y_train.loc[:, col], errors='coerce')
            for col in X_train.columns:
                X_train.loc[:, col] = pd.to_numeric(X_train.loc[:, col], errors='coerce')

            if len(df_test.columns) > 0: # then intensity coincides with those of test
                X_test = df_test[self.tensor_input_columns]
                y_test = df_test[self.tensor_output_columns]
                for col in y_test.columns:
                    y_test.loc[:, col] = pd.to_numeric(y_test.loc[:, col], errors='coerce')
                for col in X_test.columns:
                    X_test.loc[:, col] = pd.to_numeric(X_test.loc[:, col], errors='coerce')
            else:
                X_test = pd.DataFrame()
                y_test = pd.DataFrame()

            # Obtain values to save
            y_values = y_train.values
            x_values = X_train.values

            # then create tensor
            y_train = torch.tensor(y_values).float()
            X_train = torch.tensor(x_values).float()
            torch.save(X_train, self.path_train_X)
            torch.save(y_train, self.path_train_y)

            y_values = y_test.values
            x_values = X_test.values

            y_test = torch.tensor(y_values).float()
            X_test = torch.tensor(x_values).float()
            torch.save(X_test, self.path_test_X)
            torch.save(y_test, self.path_test_y)

        return (X_train, y_train, X_test, y_test)