'''
End-to-end Delay Prediction Based on Traffic Matrix Sampling.
Filip Krasniqi, Jocelyne Elias, Jeremie Leguay, Alessandro E. C. Redondi.
IEEE INFOCOM WKSHPS - NI: The 3rd International Workshop on Network Intelligence. Toronto, July 2020.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

from scipy import stats
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV
from time import time

import os
from os.path import join, expanduser

import sys
sys.path.insert(0, '../libs/')

from ns3_dataset import NS3Dataset, Scenario
from routenet_dataset import RoutenetDataset
from understanding_dataset import UnderstandingDataset

from joblib import dump

'''
Example: python rf.py cache_v1 ns3 rf_test abilene v1 all True
'''

assert len(sys.argv) == 10, "Errore" # se >= 3 va bene, altrimenti stampa Errore

arguments = sys.argv

'''
The code takes in input a space of exploration, builds a model for each combination
of hyperparams (gridsearch) and applies CV.
Training is implemented with EarlyStopping (patience hyperparam is fixed).

Input to program (eg: python train_nn_ts.py v1 ns3 v1 abilene v1 all True)
- cache_dir: dir containing last cached datasets
- dataset_id: ns3 | routenet | understanding, depending on wanted dataset
- model_dir: dir containing models and outputs
- topology: dir containing the raw dataset, usually string related to topology
- identifier: only for ns3. Defines the simulation you refer to
- intensity: only for ns3. low | all. If low, takes only intensity=0
- test_less_intensities: only for ns3. True | False. Whether to give all intensities to test or only 3 of them.

Output program: dir_model_output/<model_dir>:
- scores.txt: map of all scores for the best model
- random_forest.model: model that according to cv gave best results.
            After search, the best one is retrained.
            Search: RandomizedSearchCV
            CV: KFold
'''

'''
Initialize variables obtained in input and globals
'''
cache_dir = str(arguments[1])
dataset_name = str(arguments[2])
model_dir = str(arguments[3])
topology= str(arguments[4])
identifier = "simulation_{}".format(str(arguments[5]))
test_less_intensities = bool(arguments[7] == "True")
scenario = int(arguments[8])
assert scenario >= 1 and scenario <= 3, "Wrong value describing the scenario"
scenario = [Scenario.LEVEL_1, Scenario.LEVEL_2, Scenario.LEVEL_3][scenario-1]
scenario = [Scenario.LEVEL_1, Scenario.LEVEL_2, Scenario.LEVEL_3][scenario-1]
num_threads = 1

# where to save model and scores to file
base_dir_proj = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29'])
dir_model_exported = join(*[base_dir_proj, "exported"])
dir_model_output = join(*[dir_model_exported, "crossvalidation"])
dir_output = join(*[dir_model_output, model_dir])+"/"

if(not os.path.isdir(dir_model_exported)):
    os.mkdir(dir_model_exported)

if(not os.path.isdir(dir_model_output)):
    os.mkdir(dir_model_output)

if(not os.path.isdir(dir_output)):
    os.mkdir(dir_output)

use_ns3 = only_low = use_routenet = use_understanding = False
if dataset_name == "understanding":
    use_understanding = True
elif dataset_name == "routenet":
    use_routenet = True
else:
    use_ns3 = True
    if len(sys.argv) > 6:
        dataset_intensities = str(arguments[6])
        if dataset_intensities == "low":
            only_low = True

'''
Import dataset in form of dataframes. Result: dataset_container.{dfs_e2e_train, dfs_e2e_test}
Split to CV is done by the RandomizedSearchCV
'''
if use_ns3:
    coefficient_delay = 1
    if "ms" in cache_dir:
        coefficient_delay = 1000
        print("INFO: Computing delay in ms...")
    dataset_container = NS3Dataset(scenario = scenario, generate_tensors = False, test_less_intensities = test_less_intensities, only_low = only_low, cache_dir = cache_dir, topology = topology, identifier = identifier)
    dataset_origin = 'ns3'
elif use_routenet:
    dataset_container = RoutenetDataset(topology = topology, cache_dir = cache_dir, generate_tensors = False)
    dataset_origin = 'routenet'
elif use_understanding:
    dataset_container = UnderstandingDataset(topology = topology, cache_dir = cache_dir, generate_tensors = False)
    dataset_origin = 'understanding'
else:
    dataset_container = TestDataset(topology = topology, cache_dir = cache_dir, generate_tensors = True)
    dataset_origin = 'test_dataset'

print("INFO: using {} threads".format(num_threads))
model_dir += "_{}".format(dataset_origin)
print("INFO: model dir is {}".format(model_dir))

# remove traffic / capacity related to self loops
prefixes_cols_self_edges = ["{}_{}".format(i, i) for i in range(dataset_container.max_num_nodes)]
prefixes_cols_edges = dataset_container.get_edge_prefixes()

# define which and select input features
only_mean_std = True
use_capacities = True
if not use_capacities:
    print("WARNING: forcing not to use capacities in input!")
also_capacity = (scenario == Scenario.LEVEL_2 or scenario == Scenario.LEVEL_3) and use_capacities

if use_ns3:
    if only_mean_std:
        # among the extracted features, only select mean and std (i.e., don't consider quantiles)
        print("WARNING: only considering mean and std as extracted features")
        if also_capacity:
            input_columns = [col for col in dataset_container.input_columns if ("traffic" in col and ("mean" in col or "std" in col) and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges)) or ("capacity" in col and any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_edges))]
        else:
            input_columns = [col for col in dataset_container.input_columns if ("traffic" in col and ("mean" in col or "std" in col) and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges))]
    else:
        if also_capacity:
            input_columns = [col for col in dataset_container.input_columns if ("traffic" in col and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges)) or ("capacity" in col and any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_edges))]
        else:
            input_columns = [col for col in dataset_container.input_columns if ("traffic" in col and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges))]

    if any("capacity" in col for col in input_columns):
        print("INFO: using also capacities in input!")
    else:
        print("WARNING: not using capacity in input!")
else:
    input_columns = dataset_container.input_columns

output_columns = [col for col in dataset_container.dfs_e2e_train.columns if "delay_e2e_" in col and "mean" in col]

shuffle = True
if not shuffle:
    print("WARNING: not shuffling!")
else:
    dataset_container.dfs_e2e_train = dataset_container.dfs_e2e_train.sample(frac=1).reset_index(drop=True)

print("Data: completed. df_train shape: {}, df_test shape: {}".format(dataset_container.dfs_e2e_train.shape, dataset_container.dfs_e2e_test.shape))

X_train = dataset_container.dfs_e2e_train[input_columns]
y_train = dataset_container.dfs_e2e_train.loc[:, output_columns].dropna().multiply(10**3)
X_test = dataset_container.dfs_e2e_test[input_columns]
y_test = dataset_container.dfs_e2e_test.loc[:, output_columns].dropna().multiply(10**3)

# routenet: sets output to -1 when no OD flow, for consistency I always set it to 0
possible_negative_output_cols = [col for i, col in enumerate(output_columns) if int(i / dataset_container.max_num_nodes) == i % dataset_container.max_num_nodes]
del dataset_container
y_train[possible_negative_output_cols] = 0
y_test[possible_negative_output_cols] = 0

do_sampling = False
if do_sampling:
    print("WARNING: sampling dataset")
    sampling_rate = 1.0
    new_size = int(X_train.shape[0] * sampling_rate)
    indices = np.random.choice(X_train.shape[0], new_size, replace=False)
    X_train = X_train.loc[indices,:].dropna()
    y_train = y_train.loc[indices,:].dropna()

    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=64, batch_size=512)
    X_train = ipca.fit_transform(X_train)

    # X_train = X_train.reshape(X_train.shape[1], -1)
    X_test = ipca.transform(X_test)

    pca_path = '{}/ipca.joblib'.format(dir_output)
    dump(ipca, pca_path)

print("INFO: TRAIN INPUT SHAPE: {}".format(X_train.shape))
print("INFO: TRAIN OUTPUT SHAPE: {}".format(y_train.shape))


# build a classifier
regr = RandomForestRegressor(random_state = 0, n_jobs = num_threads)

# Report best scores at the end of execution
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# specify parameters and distributions to sample from
n_iter_search = 1
cv_k = 3
if n_iter_search == 1:
    print("INFO: only one search!")
    param_dist = {"max_depth": [60],
                "max_features": [153],
                "min_samples_split": [6],
                "bootstrap": [True],
                "criterion": ["mse"],
                "n_estimators": [63]}
else:
    if len(input_columns) <= 100:
        max_features = stats.randint(32, 96)
    elif len(input_columns) < 150:
        max_features = stats.randint(64, 128)
    else:
        max_features = stats.randint(64, 168)
    print("INFO: searching in the exploration space with size {}".format(n_iter_search))
    param_dist = {"max_depth": stats.randint(32, 48),
                  "max_features": max_features,
                  "min_samples_split": stats.randint(8, 12),
                  "bootstrap": [True],
                  "criterion": ["mse"],
                  "n_estimators": stats.randint(32, 48)}

# define and executes RandomizedSearchCV
random_search = RandomizedSearchCV(regr, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv = KFold(n_splits=cv_k, shuffle=True), iid=False, n_jobs = num_threads)
start = time()
random_search.fit(X_train, y_train)
print("INFO: RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

# score the best model
model = random_search.best_estimator_

print("SUCCESS: computing scores for best model")

prediction_train = model.predict(X_train)
prediction_test = model.predict(X_test)

del X_train, X_test

mse_train = mean_squared_error(y_train, prediction_train)
mse_test = mean_squared_error(y_test, prediction_test)

r2_train = r2_score(y_train, prediction_train)
r2_test = r2_score(y_test, prediction_test)

mae_train = mean_absolute_error(y_train, prediction_train)
mae_test = mean_absolute_error(y_test, prediction_test)

# evs_train = explained_variance_score(y_train, prediction_train)
# evs_test = explained_variance_score(y_test, prediction_test)

del y_train, y_test

print("INFO: MSE: Train = {}, Test = {}".format(mse_train, mse_test))
print("INFO: MAE: Train = {}, Test = {}".format(mae_train, mae_test))
# print("INFO: Explained variance: Train = {}, Test = {}".format(evs_train, evs_test))
print("INFO: R2: Train = {}, Test = {}".format(r2_train, r2_test))

scores_path = '{}/scores.txt'.format(dir_output)
f = open(scores_path, "w")
# f.write(str({"MSE": {"Train":mse_train, "Test":mse_test}, "MAE": {"Train":mae_train, "Test":mae_test}, "EVS": {"Train":evs_train, "Test":evs_test}, "R2": {"Train":r2_train, "Test":r2_test}}))
f.write(str({"MSE": {"Train":mse_train, "Test":mse_test}, "MAE": {"Train":mae_train, "Test":mae_test}, "R2": {"Train":r2_train, "Test":r2_test}}))
f.close()

model_path_joblib = '{}/random_forest.joblib'.format(dir_output)
dump(model, model_path_joblib)

print("SUCCESS: best model at {}".format(model_path_joblib))
print("SUCCESS: scores at {}".format(scores_path))
