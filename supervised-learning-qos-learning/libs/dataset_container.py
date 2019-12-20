from columns import *

from os import listdir, scandir
from os.path import isfile, join,expanduser, isdir
import pandas as pd
pd.set_option('precision', 9)
from sklearn import preprocessing
import pathlib

import torch
torch.set_printoptions(precision=9)
import scipy.sparse as sp
from dataset_lstm import Dataset as DatasetSequence
from dataset import Dataset
from torch.utils.data import DataLoader
import itertools
from abc import ABCMeta, abstractmethod

'''
Abstract class defining the default procedure to apply when instancing a dataset.
v0.1 accepts three subclasses, that are, RoutenetDataset, UnderstandingDataset, NS3Dataset.
init: initialize class variables and procedes with the general approach to
initialize the dataset, that is the following:
- init directory variables
- init information regarding the simulation, i.e, num_periods and num_nodes
- init variables related to the subclass (need overriding of init_variables_dataset)
- computes, if required, adjacency and spectral convolution matrix
- initialize columns for raw data, dataframes (i.e., input for RF), tensors
- initialize files related to cache
- builds (from cache or raw data) dataframes and eventually write them to cache
- builds tensors (from cache or raw data)

read_from_cache: False to force the building of the cache regardless from the fact that it already exists
visualization: True to avoid to read all the cache because the class will be used only to access raw data with get_dataframe_visualization
max_num_nodes: same as num_nodes for now, used to generalize in case of next implementation with general number of nodes
subsets_approximate_fraction: during CV for tensors, approximate size of validation set
only_test: set to True in case training is not required (faster)
do_normalization: True if you want to normalize the dataset at this point
do_capping: True if you want to cap the dataset
impute_traffic: whether to impute also traffic. Delay is imputed by default.
    Imputation is not so valuable anymore, as it rarely happens to have 0 values for delay or traffic.
cache_dir: dir name containing the cache
generate_tensors: True if you want only the tensors, False if you want only the dataframes.
    Not considered in case of building the cache.
build_adjacency_matrix: True if you want to initialize the variables related to adjacency
'''
class DatasetContainer():
    def __init__(self,
                topology = "abilene",
                read_from_cache = True,
                visualization = False,
                max_num_nodes = -1,
                subsets_approximate_fraction = 0.2,
                only_test = False,
                do_normalization = False,
                do_capping = False,
                impute_traffic = True,
                cache_dir = "cache_T",
                generate_tensors = False,
                build_adjacency_matrix = False,
                use_PCA = False,
                **kwargs):
        'Initialize dataset container'
        self.only_test = only_test
        self.topology = topology
        self.impute_traffic = impute_traffic
        self.do_normalization = do_normalization
        self.do_capping = do_capping
        self.generate_tensors = generate_tensors
        self.read_from_cache = read_from_cache
        self.use_PCA = use_PCA

        self.max_num_nodes = max_num_nodes
        self.subsets_approximate_fraction = subsets_approximate_fraction

        '''
        Directories
        '''
        self.init_base_directory()
        self.num_periods, self.num_nodes = self.get_info_simulation()

        if self.max_num_nodes < self.num_nodes:
            self.max_num_nodes = self.num_nodes

        self.init_variables_dataset()

        if build_adjacency_matrix:
            # build adjacency matrix from one simulation. Assume adjacency is the same for all the environment
            # first: sparse matrix. Second: tensor representing the sparse
            self.adjacency, self.adjacency_sparse = self.compute_adjacency()

            # compute the degree matrix from the adjacency
            self.A_hat, self.sp_inv_D = self.get_spectral_conv_matrix()

        # initialize columns
        self.init_columns()

        # files that contains cached datasets, both for tensors and dataframes
        self.cache_dir = cache_dir
        self.init_cachefiles()

        if not visualization:
            # given maps, I need to concatenate dataframes, normalize them, and build the mapped tensors
            (self.dfs_e2e_train, self.dfs_e2e_test) = self.aggregateDataframes()

            # given entire datasets, build the tensors
            if self.generate_tensors:
                (self.X_train, self.y_train, self.X_test, self.y_test) = self.getTensorsFromDataframes()
            else:
                print("WARNING: not generating tensors!")


        self.processed_train_validate = self.folder_cache + "/pyg_train.dataset"
        self.processed_test = self.folder_cache + "/pyg_test.dataset"

    '''
    Initializes variables related to inheriting classes
    Override to initialize useful variables related to the specific overriding
    if more variables are needed before the actual reading phase
    '''
    @abstractmethod
    def init_variables_dataset(self):
        pass

    '''
    Initializes variables related to base directory, i.e., root.
    dir_datasets: base directory of all the topologies. Identified by a dataset implementation.
    dir_datasets_env: directory of the specific environment. Identified by what is
        unique in the considered dataset implementation.
    '''
    @abstractmethod
    def init_base_directory(self):
        pass

    '''
    Initializes cache files. They are 6: df_{train, test}.txt + {X,Y}_{train, test}.txt.
    First two are the dataframes, other 4 relates to the pytorch tensors.
    '''
    def init_cachefiles(self):
        if self.folder_cache is None:
            raise ValueError("ERROR: cache not specified")
        if(not isdir(self.folder_cache)):
            pathlib.Path(self.folder_cache).mkdir(parents=True, exist_ok=True)
        self.path_train = join(*[self.folder_cache, "df_train.txt"])
        self.path_test = join(*[self.folder_cache, "df_test.txt"])
        self.path_train_X = join(*[self.folder_cache, "X_train.txt"])
        self.path_train_y = join(*[self.folder_cache, "y_train.txt"])
        self.path_test_X = join(*[self.folder_cache, "X_test.txt"])
        self.path_test_y = join(*[self.folder_cache, "y_test.txt"])

    '''
    Initialize columns. In the abstract case, I define only traffic, delay, capacity.
    They may also change depending on the dataset implementation.
    '''
    def init_columns(self):
        self.traffic_cols = list(build_columns_only_traffic(self.num_nodes))
        self.delay_cols = list(build_columns_only_delay_e2e(self.num_nodes))
        self.capacity_cols = list(build_columns_capacity(self.num_nodes))
        # TODO capacity_cols: only those of links
        self.load_cols = list(build_columns_only_load(self.num_nodes))
        self.dropped_cols = list(build_columns_only_dropped(self.num_nodes))

    '''
    Returns (num_periods, num_nodes) assuming self.dir_datasets_env
    '''
    @abstractmethod
    def get_info_simulation(self):
        pass

    '''
    Aggregate map of dataframes into two different: train, test.
    Specifically, it reads all the raw data, extract the features and creates
    the dataframes from these.
    '''
    @abstractmethod
    def aggregateDataframes(self):
        pass

    '''
    Assuming cache already exists for the dataframes, returns the pair (df_train, df_test)
    '''
    def init_dataframes_from_cache(self, is_numpy = False):
        print("INFO: reading dataframes from files")
        columns = np.append(self.input_columns, self.output_columns)
        df_train = None
        if not self.only_test:
            if is_numpy:
                df_train = np.loadtxt(self.path_train, dtype=np.ndarray)
            else:
                try:
                    df_train = pd.read_csv(self.path_train, sep=" ", header=None, names=list(columns), index_col=False)
                except:
                    print("WARNING: obsolete. Columns are all those ...")
                    input_columns = self.feature_cols("mean", remove_self_loops=False) + self.feature_cols("std",
                                                                                                           remove_self_loops=False) + self.capacity_cols
                    output_columns = self.feature_cols("mean", build_columns_only_delay_e2e,
                                                       remove_self_loops=False) + self.feature_cols("std",
                                                                                                    build_columns_only_delay_e2e,
                                                                                                    remove_self_loops=False)
                    columns = np.append(input_columns, output_columns + ["intensity", "simulation"])
                    df_train = pd.read_csv(self.path_train, sep=" ", header=None, names=list(columns), index_col=False)

        if is_numpy:
            df_test = np.loadtxt(self.path_test, dtype=np.ndarray)
        else:
            try:
                df_test = pd.read_csv(self.path_test, sep=" ", header=None, names=list(columns),index_col=False)
            except:
                print("WARNING: obsolete. Columns are all those ...")
                input_columns = self.feature_cols("mean", remove_self_loops=False) + self.feature_cols("std", remove_self_loops=False) + self.capacity_cols
                output_columns = self.feature_cols("mean", build_columns_only_delay_e2e, remove_self_loops=False) + self.feature_cols("std", build_columns_only_delay_e2e, remove_self_loops=False)
                columns = np.append(input_columns, output_columns + ["intensity", "simulation"])
                df_test = pd.read_csv(self.path_test, sep=" ", header=None, names=list(columns), index_col=False)
        return df_train, df_test

    '''
    Computes the convolution matrix as described here: https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0
    '''
    def get_spectral_conv_matrix(self, topology = None):
        adjacency, _ = self.compute_adjacency(topology)
        degree_matrix = self.get_degree_matrix_from_file(topology)

        D_inv = degree_matrix**-0.5
        for i in range(D_inv.shape[0] ** 2):
            row = int(i/D_inv.shape[0])
            col = i%D_inv.shape[0]
            if row != col:
                D_inv[row][col] = 0

        A_hat = D_inv * adjacency * D_inv
        A_hat = torch.Tensor(A_hat)
        sparse_sp_degree = sp.coo_matrix(D_inv, shape=(1, self.max_num_nodes, self.max_num_nodes), dtype=np.float32)
        sparse_torch_degree = self.sparse_mx_to_torch_sparse_tensor(sparse_sp_degree)
        return A_hat, sparse_torch_degree

    '''
    Compute and return the (diagonal) degree matrix from the adjacency given a topology.
    '''
    def get_degree_matrix_from_file(self, topology = None):
        adjacency = self.get_adjacency_from_file(topology)
        degree_matrix = np.reshape(np.zeros(self.max_num_nodes**2), (self.max_num_nodes, -1))
        indices = [j for j, x in enumerate(adjacency) if x != 0]

        for idx in indices:
            first_node = int(idx / self.max_num_nodes)
            second_node = idx % self.max_num_nodes
            degree_matrix[first_node][first_node] += 1
            degree_matrix[second_node][second_node] += 1

        return degree_matrix

    '''
    Computes the adjacency matrix adapting it to the learning.
    The sparse tensor is the one used for learning.
    '''
    def compute_adjacency(self, topology = None):
        adjacency = self.get_adjacency_from_file(topology)
        adjacency = np.reshape(adjacency, (self.max_num_nodes, self.max_num_nodes))

        adjacency = sp.coo_matrix(adjacency,
                        shape=(1, self.max_num_nodes, self.max_num_nodes),
                        dtype=np.float32)

        # make it symmetric
        adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency) - adjacency.multiply(adjacency.T > adjacency)

        # add diagonal
        adjacency = adjacency + sp.eye(adjacency.shape[0])  # by adding diagonal we handle the problem of having at least one value in each row

        # normalize and return
        adjacency = self.normalize(adjacency)
        return adjacency, self.sparse_mx_to_torch_sparse_tensor(adjacency)

    '''
    Given a topology, return prefixes <node_1>_<node_2> related to an edge
    '''
    def get_edge_prefixes(self):
        filename_links = join(*[self.dir_datasets_env, "links.txt"])
        # fill sparse matrix with values of adjacency matrix
        links_f = open(filename_links , 'r' )
        prefixes = []
        for count_line, line in enumerate(links_f.readlines()):
            if(count_line == 0):
                self.num_links = int(line)
            else:
                links_pair = line.split(" ")
                prefixes.append("{}_{}".format(int(links_pair[0]), int(links_pair[1])))
        return prefixes

    '''
    Given a topology, computes the adjacency matrix from the links file
    '''
    def get_adjacency_from_file(self, topology = None):
        if topology is None:
            topology = self.topology
        filename_links = join(*[self.dir_datasets_env, "links.txt"])
        adjacency = np.zeros(self.max_num_nodes**2)
        # fill sparse matrix with values of adjacency matrix
        links_f = open(filename_links , 'r' )
        for count_line, line in enumerate(links_f.readlines()):
            if(count_line == 0):
                self.num_links = int(line)
            else:
                links_pair = line.split(" ")
                first_node = int(links_pair[0])
                second_node = int(links_pair[1])
                adjacency[first_node*self.num_nodes+second_node] = 1

        links_f.close()

        return adjacency

    """
    Return adjacency matrix expressed as coo
    """
    def coo_from_adj(self):
        adjacency = self.get_adjacency_from_file()
        source = []
        target = []

        for j in range(self.max_num_nodes ** 2):
            s = int(j/self.max_num_nodes)
            t = j % self.max_num_nodes
            if adjacency[j] > 0:
                source.append(s)
                target.append(t)

        return source, target

    '''
    Convert a scipy sparse matrix to a torch sparse tensor.
    '''
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    '''
    Row-normalize sparse matrix
    '''
    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    '''
    Computes the imputation by assigning the closest non-0 value to 0 value.
    Implemented to work, by default, on e2e delay.
    Change params to use it for other purposes
    '''
    def imputing(self, df, prefix_blacklist = "delay_e2e"):
        blacklist_columns = ["{}_{}_{}".format(prefix_blacklist, i,i) for i in range(self.num_nodes)] # list of columns on which we expect 0

        # handle missing values with following temporary solution.
        '''
        Update all values on which we have 0 for non inner loops. In those cases we find the closest
        value by considering as a similarity the epoch
        '''
        for col in df.columns:
            if col not in blacklist_columns:
                mask = df[col] == 0
                mask_correct_values = df[col] != 0
                count_MV = 0
                for m in mask:
                    if m:
                        count_MV +=1
                if count_MV > 0:
                    # compute closest considering epoch
                    # build parallel array to mask on which we define an array of same size of mask but with associated
                    imputation_values = []
                    imputation_indices = []
                    for i, is_MV in enumerate(mask):
                        if is_MV:
                            # find value in mask_correct_values closest to i
                            vector_distances = []
                            max_val = len(mask_correct_values)
                            for j, is_CV in enumerate(mask_correct_values):
                                if is_CV:
                                    value = abs(j - i)
                                else:
                                    value = max_val
                                vector_distances.append(value)
                            index_mask_correctval = np.argmin(vector_distances)
                            imputation_values.append(df.loc[index_mask_correctval, col])
                            imputation_indices.append(index_mask_correctval)

                    df.loc[mask, col] = imputation_values

        return df

    '''
    Obtain dataframe given correct kwargs related to the class
    '''
    @abstractmethod
    def getDataframeFromSimulation(self, **kwargs):
        pass

    '''
    Returns a dataframe for visualization. It will be a subset of the real one
    '''
    @abstractmethod
    def get_dataframe_visualization(self, **kwargs):
        pass

    '''
    Given a dataframe and his continuous columns, normalize it
    '''
    def getNormalizedDataframe(self,df,continuous_columns = None):
        if continuous_columns is None:
            continuous_columns = self.traffic_cols
        scaler = preprocessing.StandardScaler()
        df.loc[:, continuous_columns]=scaler.fit_transform(df[continuous_columns])
        df.loc[:, continuous_columns]=scaler.fit_transform(df[continuous_columns])
        return df

    '''
    Given a Dataframe, it caps it, i.e., removes outliers by setting those higher than 0.01 and 0.99 to the closer threshold
    '''
    def getCappedDataframe(self,df, to_cap_columns, quantiles = [0.01,0.99]):
        for col in df.columns:
            if col in to_cap_columns:
                percentiles = df[col].quantile(quantiles).values
                indices_low_cap = df[col] <= percentiles[0]
                indices_high_cap = df[col] >= percentiles[1]
                df.loc[indices_low_cap, col] = percentiles[0]
                df.loc[indices_high_cap, col] = percentiles[1]
        return df

    def need_init_tensors(self):
        return self.read_from_cache and isfile(self.path_train_X) and isfile(self.path_train_y) and isfile(self.path_test_X) and isfile(self.path_test_y)

    '''
    Given the pre-initialized dataframes, builds the tensors for pytorch.
    Filters the columns of the dataframes by checking whether any string in
    filter_columns is part of the column.
    '''
    def getTensorsFromDataframes(self, features_pca = 128):
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

            X_train = self.dfs_e2e_train.loc[:, self.tensor_input_columns]
            y_train = self.dfs_e2e_train.loc[:, self.tensor_output_columns]

            for col in y_train.columns:
                y_train.loc[:, col] = pd.to_numeric(y_train.loc[:, col], errors='coerce')
            for col in X_train.columns:
                X_train.loc[:, col] = pd.to_numeric(X_train.loc[:, col], errors='coerce')

            if len(self.dfs_e2e_test.columns) > 0: # then intensity coincides with those of test
                X_test = self.dfs_e2e_test.loc[:, self.tensor_input_columns]
                y_test = self.dfs_e2e_test.loc[:, self.tensor_output_columns]
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

        if self.use_PCA:
            if not self.only_test:
                X_train = self.PCA(X_train, features_pca)
            X_test = self.PCA(X_test, features_pca)

        return (X_train, y_train, X_test, y_test)

    """
    Compute PCA given tensors in input
    """
    def PCA(self, X, k):
        X_mean = torch.mean(X, 0)
        X = X - X_mean.expand_as(X)
        # svd
        U, S, V = torch.svd(torch.t(X))
        # TODO solve problem here. Too big instance of RAM
        return torch.mm(X, U[:, :k])

    '''
    Given a tensor and an array of indices of columns, returns the tensor filtered
    on the defined columns
    '''
    def filter_tensor_by_column(self, tensor, col_indices):
        gather_matrix = []
        for _ in range(tensor.size(0)):
            gather_matrix.append(col_indices)
        return torch.gather(tensor, 1, torch.tensor(gather_matrix, dtype=torch.long))

    '''
    Return test according to current chosen range of columns and simulations
    '''
    def get_test(self, range_columns = None):
        range_columns_output = list(range(self.max_num_nodes ** 2))
        if range_columns is None:
            range_columns = list(range(self.X_test.size(1)))
        return self.filter_tensor_by_column(self.X_test, range_columns), self.filter_tensor_by_column(self.y_test, range_columns_output)

    '''
    Return train according to current chosen range of columns and simulations
    '''
    def get_train(self, range_columns = None):
        range_columns_output = list(range(self.max_num_nodes ** 2))
        if range_columns is None:
            # default: only TM
            range_columns = list(range(self.X_train.size(1)))
        return self.filter_tensor_by_column(self.X_train, range_columns), self.filter_tensor_by_column(self.y_train, range_columns_output)

    '''
    Obtain tensors that are all merged together
    '''
    def shuffle_all_tensors_together(self, range_columns = None, range_output_columns = None):
        # takes X_train, X_test and merges them together. Same for y. Then shuffles them.
        self.X = torch.cat([self.X_train, self.X_test])
        self.y = torch.cat([self.y_train, self.y_test])

        # create perm if not present, otherwise read them
        path_permutations = join(*[self.folder_cache, "permutations.txt"])
        if isfile(path_permutations):
            print("INFO: reading permutations...")
            perm = torch.load(path_permutations)
        else:
            print("INFO: creating permutations...")
            perm = torch.randperm(self.X_train.size(0) + self.X_test.size(0))
            print("INFO: writing permutations...")
            torch.save(perm, path_permutations)
        self.X = self.X[perm]
        self.y = self.y[perm]

        self.X_train, self.y_train = self.X[0:self.X_train.size(0),:], self.y[0:self.y_train.size(0),:]
        shift_test = self.X_train.size(0)
        self.X_test, self.y_test = self.X[shift_test:shift_test + self.X_test.size(0),:], self.y[shift_test:shift_test + self.y_test.size(0),:]

    '''
    Initializes values of train + validate and indices to have CV given a k value
    '''
    def init_cv(self, cv_k, range_columns = None, range_output_columns = None):
        print("INFO: starting CV split")

        if range_columns is None:
            range_columns = list(range(self.X_train.size(1)))
        if range_output_columns is None:
            range_output_columns = list(range(self.max_num_nodes ** 2))
        # consider train and validate together
        self.current_cv_k = cv_k
        print("INFO: final shape of train dataset: {}".format(self.X_train.shape))

        if self.subsets_approximate_fraction > 1:
            print("ERROR: subsets_approximate_fraction is a fraction, thus must be < 1")

        subsets_approximate_size = int(self.X_train.shape[0] * self.subsets_approximate_fraction)
        num_subsets = int(self.X_train.shape[0]/subsets_approximate_size) # last one: add it to testing

        print("INFO: #subsets: {}".format(num_subsets))
        all_indices = np.random.choice(self.X_train.shape[0], self.X_train.shape[0], replace=False)
        self.indices = np.array_split(all_indices, num_subsets) # split subset in approximately sizes of subsets_approximate_size
        self.validation_indices = np.random.choice(num_subsets, cv_k, replace=False)

    '''
    Function loading dataset to handle batch learning.
    In this case it differs from the training case with fixed hyperparams, as we have the indices
    previously computed to allow CV.
    Shuffle only when training.
    Test does not require this, as we compute mse in our way.
    '''
    def get_data_loaders(self, batch_size, indices_train, indices_validate, window_size = None):
        if window_size is None:
            # no window size means we don't provide the data as time series
            train_set = Dataset(self.X_train[indices_train], self.y_train[indices_train])
            validation_set = Dataset(self.X_train[indices_validate], self.y_train[indices_validate])
        else:
            # window size means to provide the data as sequence
            train_set = DatasetSequence(self.X_train[indices_train], self.y_train[indices_train], window_size)
            validation_set = DatasetSequence(self.X_train[indices_validate], self.y_train[indices_validate], window_size)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=self.X_train[indices_validate].shape[0], shuffle=False)
        return train_loader, val_loader

    '''
    Returns the data loaders given the current k and the batch size.
    Requirement: call init_cv before
    '''
    def get_current_data_loaders(self, current_k, batch_size, window_size = None):
        assert self.indices is not None and self.validation_indices is not None and self.X_train is not None and self.y_train is not None and current_k < self.current_cv_k, "Something is wrong"
        indices_train = np.array([ elem for i, singleList in enumerate(self.indices) if i != self.validation_indices[current_k] for elem in singleList])
        indices_validate = np.array(self.indices[self.validation_indices[current_k]])
        return self.get_data_loaders(batch_size, indices_train, indices_validate, window_size)

    def get_test_data_loader(self, window_size = None):
        if window_size is None:
            # no window size means we don't provide the data as time series
            test_set = Dataset(self.X_test, self.y_test)
        else:
            # window size means to provide the data as sequence
            test_set = DatasetSequence(self.X_test, self.y_test, window_size)

        test_loader = DataLoader(test_set, batch_size=self.X_test.size(0), shuffle=True)
        return test_loader
