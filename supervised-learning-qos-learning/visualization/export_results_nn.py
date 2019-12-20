import itertools
from os.path import isfile, join,expanduser, isdir

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0)
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(12,8.5)})
sns.set(font_scale=1.5)

import sys
sys.path.insert(0, '../libs/')
sys.path.insert(0, '../pytorch/')
from columns import *

import os
from os.path import join, expanduser

import torch

from ns3_dataset import NS3Dataset, Scenario
from routenet_dataset import RoutenetDataset
from understanding_dataset import UnderstandingDataset

plt.show(block=True)
plt.interactive(False)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#L3:
# arguments = ["rf.py", "cache_v2", "ns3", "nn_search_v1_L3", "abilene", "v1_L3", "all", False, 3]
#L2:
# arguments = ["rf.py", "cache_v2", "ns3", "nn_search_v1_L2", "abilene", "v1_L2", "all", False, 2]
#L1:
# arguments = ["rf.py", "cache_v2", "ns3", "nn_search_v1_L1", "abilene", "v1_L1", "all", False, 1]
arguments = sys.argv
cache_dir = str(arguments[1])
dataset_id = str(arguments[2])
model_dir = str(arguments[3])
topology = str(arguments[4])
identifier = "simulation_{}".format(str(arguments[5]))
test_less_intensities = bool(arguments[7] == "True")
do_normalization = False
log_epoch = False
threshold_log = 100
datasets_output = "datasets"
shuffle_all_tensors_together = False
real_case_split = False
num_threads = 16
torch.set_num_threads(num_threads)
scenario = int(arguments[8])
assert scenario >= 1 and scenario <= 3, "Wrong value describing the scenario"
scenario = [Scenario.LEVEL_1, Scenario.LEVEL_2, Scenario.LEVEL_3][scenario-1]
only_test = True

use_ns3 = only_low = use_routenet = use_understanding = False
if dataset_id == "understanding":
    use_understanding = True
elif dataset_id == "routenet":
    use_routenet = True
else:
    use_ns3 = True
    if len(sys.argv) > 6:
        dataset_intensities = str(arguments[6])
        if dataset_intensities == "low":
            only_low = True

if use_ns3:
    coefficient_delay = 1
    if "ms" in cache_dir:
        coefficient_delay = 1000
        print("INFO: Computing delay in ms...")
    """"""
    dataset_container = NS3Dataset(also_pyg=False, scenario=scenario, generate_tensors=True,
                                                  test_less_intensities=test_less_intensities, only_low=only_low,
                                                  cache_dir=cache_dir, topology=topology, identifier=identifier)
    dataset_origin = 'ns3'
elif use_routenet:
    dataset_container = RoutenetDataset()
    dataset_origin = 'routenet'
elif use_understanding:
    dataset_container = UnderstandingDataset()
    dataset_origin = 'understanding'
else:
    dataset_container = TestDataset()
    dataset_origin = 'test_dataset'

dir_log_output = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', 'exported', 'crossvalidation', "results"])
dir_model_output = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', "exported", "crossvalidation"])
path_output = join(*[dir_log_output, model_dir])
path_boxplot = join(*[path_output, "boxplot.csv"])
if (not os.path.isdir(dir_log_output)):
    os.mkdir(dir_log_output)

if (not os.path.isdir(path_output)):
    os.mkdir(path_output)

if (not os.path.isdir(dir_model_output)):
    raise ValueError("Model directory does not exist")

# import nn
env_name = "{}_{}".format(model_dir, dataset_id)
model_name = "NN.model"
env_path = join(*[dir_log_output, env_name])
model_path = join(*[dir_model_output, env_name, model_name])
net = torch.load(model_path)["model"]
print(net)

def columns_od(name, remove_self_loops = True):
    if remove_self_loops:
        return ["{}_{}_{}".format(name, int(i/num_nodes), i%num_nodes) for i in range(num_nodes ** 2) if i%num_nodes != int(i / num_nodes)]
    else:
        return ["{}_{}_{}".format(name, int(i / num_nodes), i % num_nodes) for i in range(num_nodes ** 2)]

"""
Function returning test dataset. By default it returns it as a data loader
"""
def get_test(test_as_dataloader = True):
    if test_as_dataloader:
        test_loader = dataset_container.get_test_data_loader()

        count_test_instances = 0
        for instance in test_loader:
            if count_test_instances > 0:
                print("WARNING: test loader is batched")
            X_test, y_test = instance
            count_test_instances += 1
    else:
        X_test, y_test = dataset_container.X_test, dataset_container.y_test

    return X_test, y_test

# TODO keep going with infering
X_test, real_values = get_test(False)

predictions = net(X_test).mul(0.001)
errors = torch.abs(torch.sub(predictions, real_values)).detach().numpy()
real_values = real_values.detach().numpy()
predictions = predictions.detach().numpy()

num_nodes = 12
columns_error, columns_delay, columns_predicted = columns_od("abs_error", remove_self_loops = True), columns_od("target", remove_self_loops = True), columns_od("predicted", remove_self_loops = True)
df_error = pd.DataFrame(columns=columns_error, data = {columns_error[i]: errors[:, i] for i in range(num_nodes ** 2 - num_nodes)})
df_delay = pd.DataFrame(columns=columns_delay, data = {columns_delay[i]: real_values[:, i] for i in range(num_nodes ** 2 - num_nodes)})
df_predicted = pd.DataFrame(columns=columns_predicted, data = {columns_predicted[i]: predictions[:, i] for i in range(num_nodes ** 2 - num_nodes)})

# regarding other parameters: intensity, simulation, capacity, ... -> I have the function that does that for me
df_other = dataset_container.get_test_iscp_dataframe()
df = pd.concat([df_error, df_delay, df_predicted, df_other], axis=1, sort=False)
len_for_intensity, len_for_capacity, len_for_pd = dataset_container.get_current_num_environments(test=True)
# order: simulation, intensity, capacity, environment
test_simulations, test_intensities, test_capacities, test_pdelays = df.Simulation.unique(), df.Intensity.unique(), \
                                                                    df.Capacity.unique(), df["P.Delay"].unique()

OD_to_check_std_rmse = "_0_9"
mse_per_intensity = False

def filter_df(df, simulation, intensity, capacity, p_delay):
    simulation_mask, intensity_mask, capacity_mask, p_delay_mask = df["Simulation"] == simulation, df["Intensity"] == intensity,\
                                                                   df["Capacity"] == capacity, df["P.Delay"] == p_delay
    total_mask = simulation_mask & intensity_mask & capacity_mask & p_delay_mask
    return df.where(total_mask).dropna()

def filter_df_sims(df, simulations):
    return df[df.Simulation.isin(simulations)]

def filter_df_col(df, cols):
    columns_to_filter = []
    for filter in cols:
        columns_current_filter = [col for col in df.columns if col.endswith(filter)]
        columns_to_filter += columns_current_filter
    columns_to_filter += list(df_other.columns.values)
    return df.loc[:, columns_to_filter]

simulations_to_show, intensities_to_show, capacities_to_show, pds_to_show = test_simulations, test_intensities,\
                                                                            test_capacities, test_pdelays
df_filtered_sim = filter_df_sims(df, simulations_to_show)
sicp = list(itertools.product(simulations_to_show, intensities_to_show,
                                                            capacities_to_show, pds_to_show))
total_length = len(sicp)

box_plot = True

if box_plot and not isfile(path_boxplot):
    rmse_values_intensity, stds_values_intensity, intensities = [], [], []
    for idx_sicp, (c_sim, c_int, c_cap, c_pd) in enumerate(sicp):
        c_sim, c_int, c_cap, c_pd = int(c_sim), int(c_int), int(c_cap), int(c_pd)
        current_df = filter_df(df, c_sim, c_int, c_cap, c_pd)
        current_targets, current_predictions = current_df.loc[:, [col for col in current_df.columns if "target" in col]], \
                                               current_df.loc[:, [col for col in current_df.columns if "predic" in col]]
        rmse = np.power(mean_squared_error(current_targets.values.flatten(), current_predictions.values.flatten()), 0.5)
        std = np.std(current_predictions.values.flatten())
        rmse_values_intensity.append(rmse)
        intensities.append(c_int)
        stds_values_intensity.append(std)
        print("INFO: S = {}/{}, I = {}/{}, C = {}/{}, P = {}/{}, Total: {}/{}.".format(c_sim + 1,
                                                                                  len(simulations_to_show),
                                                                                  c_int + 1,
                                                                                  len(intensities_to_show),
                                                                                  c_cap + 1,
                                                                                  len(capacities_to_show),
                                                                                  c_pd + 1,
                                                                                  len(pds_to_show),
                                                                                  idx_sicp + 1,
                                                                                  total_length),
              end='\r',
              flush=True)
    boxplot_current_model = pd.DataFrame(data={"RMSE": rmse_values_intensity + stds_values_intensity, "Intensity": np.concatenate((intensities, intensities)), "is_std": np.concatenate((np.repeat(1, len(intensities)), np.repeat(0, len(intensities))))})
    boxplot_current_model.to_csv(path_or_buf=path_boxplot, sep=' ', index=False, header=False)

scatter_plot = True
selected_columns = ["_3_5", "_3_4", "_4_5", "_0_3", "_0_1", "_1_3"]
if scatter_plot:
    df_filtered_col = filter_df_col(df_filtered_sim, selected_columns)
    for idx_col, col in enumerate(selected_columns):
        path_col = join(*[path_output, "scatter{}.csv".format(col)])
        df_filtered_col_current_col = filter_df_col(df_filtered_col, [col])
        avg_targets, avg_predictions, avg_abs_err, intensities = [], [], [], []
        for idx_sicp, (c_sim, c_int, c_cap, c_pd) in enumerate(sicp):
            c_sim, c_int, c_cap, c_pd = int(c_sim), int(c_int), int(c_cap), int(c_pd)
            current_df = filter_df(df, c_sim, c_int, c_cap, c_pd)
            current_targets, current_predictions = current_df.loc[:,
                                                   [col for col in current_df.columns if "target" in col]], \
                                                   current_df.loc[:,
                                                   [col for col in current_df.columns if "predic" in col]]
            avg_targets.append(np.mean(current_targets.to_numpy()))
            avg_predictions.append(np.mean(current_predictions.to_numpy()))
            avg_abs_err.append(np.mean(np.abs(current_targets.to_numpy() - current_predictions.to_numpy())))
            intensities.append(c_int)

            print("INFO: S = {}/{}, I = {}/{}, C = {}/{}, P = {}/{}, Total: {}/{}.".format(c_sim + 1,
                                                                                           len(simulations_to_show),
                                                                                           c_int + 1,
                                                                                           len(intensities_to_show),
                                                                                           c_cap + 1,
                                                                                           len(capacities_to_show),
                                                                                           c_pd + 1,
                                                                                           len(pds_to_show),
                                                                                           idx_sicp + 1,
                                                                                           total_length),
                  end='\r',
                  flush=True)

        print("SUCCESS: column {}/{} completed!".format(idx_col + 1, len(selected_columns)))

        scatter_plot_current_model = pd.DataFrame(data={"Avg.Targets": avg_targets,
                                                   "Avg.Predictions": avg_predictions,
                                                   "Avg.Abs.Errors": avg_abs_err,
                                                    "Intensity": intensities})
        scatter_plot_current_model.to_csv(path_or_buf=path_col, sep=' ', index=False)