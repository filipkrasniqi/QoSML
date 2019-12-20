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
use_PCA = False
palette = {
    "intensity_0":"#F6CE00",
    "intensity_1":"#F6AD00",
    "intensity_2":"#F6A500",
    "intensity_3":"#F67C00",
    "intensity_4":"#F66300",
    "intensity_5":"#F6530C",
    "intensity_6":"#F63E05",
    "intensity_7":"#F62A0A",
    "intensity_8":"#F62605",
    "intensity_9":"#F60000",
}

palette_idx = {
    0:"#F6CE00",
    1:"#F6AD00",
    2:"#F6A500",
    3:"#F67C00",
    4:"#F66300",
    5:"#F6530C",
    6:"#F63E05",
    7:"#F62A0A",
    8:"#F62605",
    9:"#F60000",
}

palette_intensity_val = {
    idx: palette["intensity_{}".format(idx)] for idx, _ in enumerate(palette)
}

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
                                                  cache_dir=cache_dir, topology=topology, identifier=identifier,
                                                  use_PCA=use_PCA)
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
if mse_per_intensity:
    for idx_intensity, intensity in enumerate(test_intensities):
        predictions_intensity = df.loc[:, columns_predicted].where(df["Intensity"] == intensity).dropna()
        targets_intensity = df.loc[:, columns_delay].where(df["Intensity"] == intensity).dropna()

        if OD_to_check_std_rmse is None:
            std_intensity = np.std(targets_intensity)
            rmse_intensity = np.power(mean_squared_error(predictions_intensity, targets_intensity), 0.5)
            print(intensity, mean_absolute_error(predictions_intensity, targets_intensity), std_intensity, rmse_intensity / std_intensity, std_intensity / rmse_intensity)
        else:
            # TODO questo non ha senso. Sto pensando se rimuovere questa parte...
            pred_col, target_col = [col for col in columns_predicted if OD_to_check_std_rmse in col], [col for col in columns_delay if OD_to_check_std_rmse in col]
            current_targets, current_predictions = targets_intensity.loc[:, target_col[0]], predictions_intensity.loc[:, pred_col[0]]
            std_intensity = np.std(current_targets)
            rmse_intensity = np.power(mean_squared_error(current_targets, current_predictions), 0.5)
            print(intensity, mean_absolute_error(current_targets, current_predictions), std_intensity,
                  rmse_intensity / std_intensity, std_intensity / rmse_intensity)

mse_test, mae_test, r2_test = mean_squared_error(real_values, predictions), mean_absolute_error(real_values, predictions), r2_score(real_values, predictions)

print("INFO: MSE = {}".format(mse_test))
print("INFO: MAE = {}".format(mae_test))
print("INFO: STD test = {}".format(np.std(real_values)))
print("INFO:  STD vs RMSE: {} / {} = {}".format(np.power(mse_test, 0.5), np.std(real_values), np.std(real_values) / np.power(mse_test, 0.5)))

def filter_df(df, simulation, intensity, capacity, p_delay):
    simulation_mask, intensity_mask, capacity_mask, p_delay_mask = df["Simulation"] == simulation, df["Intensity"] == intensity,\
                                                                   df["Capacity"] == capacity, df["P.Delay"] == p_delay
    total_mask = simulation_mask & intensity_mask & capacity_mask & p_delay_mask
    return df.where(total_mask).dropna()

def filter_df_sims(df, simulations):
    return df[df.Simulation.isin(simulations)]

simulations_to_show, intensities_to_show, capacities_to_show, pds_to_show = test_simulations, test_intensities,\
                                                                            test_capacities, test_pdelays
df_filtered_sim = filter_df_sims(df, simulations_to_show)

# list of whatever I want to show (indices)
dist_plot_delay = False
if dist_plot_delay:
    for c_sim, c_int, c_cap, c_pd in itertools.product(simulations_to_show, intensities_to_show,
                                                        capacities_to_show, pds_to_show):
        c_sim, c_int, c_cap, c_pd = int(c_sim), int(c_int), int(c_cap), int(c_pd)
        current_df = filter_df(df, c_sim, c_int, c_cap, c_pd)
        col = "0_9"
        delay_col = "target_{}".format(col)
        current_df = current_df.loc[:, delay_col]

        min_val = min(current_df)
        max_val = max(current_df)

        g = sns.distplot(current_df.values, color=palette_idx[c_int])
        # g = sns.scatterplot(current_df[delay_col].values, current_df[mse_col].values, color=palette[intensity])
        g.set_xlabel("Delay dist. for OD:{}, sim: {}".format(col, c_sim))
        # g.set_yticklabels(labels=[])
        # g.yaxis.set_label_position("right")

        # g.yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
        g.ticklabel_format(axis='x', style='sci', scilimits=(-3, 0))
        g.ticklabel_format(axis='y', style='sci', scilimits=(-1, 0))

        # g.xaxis.ticklabel_format(style='sci', scilimits=(1, 4))
        g.set(xlim=(min_val, max_val))
        plt.savefig("test.png")
        plt.show()

box_plot = True
if box_plot and not isfile(path_boxplot):
    rmse_values_intensity, stds_values_intensity, intensities = [], [], []
    sicp = list(itertools.product(simulations_to_show, intensities_to_show,
                                                            capacities_to_show, pds_to_show))
    total_length = len(sicp)
    # TODO calcolare ora lista di RMSE. Un punto per ogni simulation. Mostrare a dx la distribuzione della std?
    for idx_sicp, (c_sim, c_int, c_cap, c_pd) in enumerate(sicp):
        # TODO calcolare RMSE per questa simulazione
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
    g = sns.FacetGrid(boxplot_current_model, col="Intensity", hue="Intensity",
                      palette=palette_intensity_val, height=4, aspect=1.7)
    g = (g.map(sns.boxplot, x="is_std", y="RMSE"))
    # g.set(ylim=(0, 5000))
    plt.savefig("test.png")
    plt.show(block=True)
    # TODO poi salvali

dist_plot_ae = False
if dist_plot_ae:
    col = "0_9"
    delay_col = "target_{}".format(col)
    mse_col = "abs_error_{}".format(col)

    g = sns.FacetGrid(df_filtered_sim, row="Simulation", col="Intensity", hue="Intensity", palette=palette_intensity_val, height=4, aspect=1.7)
    g = (g.map(sns.distplot, mse_col))

    g.set(ylim=(0, 5000))
    plt.savefig("test.png")
    plt.show(block=True)

scatter_ae_delay = False
if scatter_ae_delay:
    col = "0_9"
    delay_col = "target_{}".format(col)
    mse_col = "abs_error_{}".format(col)
    current_df = pd.DataFrame(data={}, columns=[delay_col, mse_col, "intensity"])
    for idx_s, idx_i, idx_c, idx_p in itertools.product(simulations_to_show, intensities_idx_to_show, capacities_to_show, pds_to_show):
        i = idx_s * len_for_intensity + idx_i * len_for_capacity + idx_c * len_for_pd
        min_row, max_row = i * 350, (i+1) * 350 - 1
        intensity = int(test_intensities[idx_i].split("_")[1])
        temp_df = df.loc[min_row:max_row, [delay_col, mse_col]]
        temp_df["intensity"] = intensity
        temp_df["simulation"] = "{}_{}_{}".format(int(test_simulations[idx_s]), idx_c, idx_p)
        current_df = current_df.append(temp_df)

    g = sns.FacetGrid(current_df, row="simulation", col="intensity", hue="intensity", palette=palette_intensity_val, height=4, aspect=1.7)
    g = (g.map(sns.scatterplot, delay_col, mse_col)).add_legend()
    # g = sns.scatterplot(data=current_df, x=delay_col, y=mse_col, hue="intensity", column="simulation")
    # g.set_xlabel("Delay vs Absolute Error for OD:{}".format(col))
    # g.set_yticklabels(labels=[])
    # g.yaxis.set_label_position("right")

    # g.yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
    # g.ticklabel_format(axis='x', style='sci', scilimits=(-3, 0))
    # g.ticklabel_format(axis='y', style='sci', scilimits=(-1, 0))

    # g.xaxis.ticklabel_format(style='sci', scilimits=(1, 4))
    # g.set(xlim=(0.2375, 0.255), ylim=(-0.0001, 0.005))
    plt.savefig("test.png")
    plt.show(block=True)

plot_delays_comparison = False
if plot_delays_comparison:
    for idx_s, idx_i, idx_c, idx_p in itertools.product(simulations_to_show, intensities_idx_to_show,
                                                        capacities_to_show, pds_to_show):
        i = idx_s * len_for_intensity + idx_i * len_for_capacity + idx_c * len_for_pd
        min_row, max_row = i * 350, (i + 1) * 350 - 1
        current_df = df.loc[min_row:max_row, :]
        col = "0_9"
        intensity = test_intensities[idx_i]
        col = "0_9"

        delay_col = "target_{}".format(col)
        predicted_col = "predicted_{}".format(col)
        mse_col = "abs_error_{}".format(col)
        total_min_ae, total_max_ae = min(current_df[mse_col].values), max(current_df[mse_col].values)
        min_val = min(current_df[delay_col].values)
        max_val = max(current_df[delay_col].values)

        delta = abs(total_max_ae - total_min_ae) / 10
        min_val_y = total_min_ae - delta
        max_val_y = total_max_ae + delta

        g = sns.lineplot(current_df[delay_col].index, current_df[delay_col].values, color=palette[intensity])
        g = sns.lineplot(current_df[delay_col].index, current_df[predicted_col].values, color="blue")
        g.set_xlabel("Delay vs Absolute Error for OD:{}".format(",".join(col.split("_")[3:5])))
        # g.set_yticklabels(labels=[])
        # g.yaxis.set_label_position("right")

        # g.yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
        # g.ticklabel_format(axis='x', style='sci', scilimits=(-3, 0))
        g.ticklabel_format(axis='y', style='sci', scilimits=(-1, 0))

        # g.xaxis.ticklabel_format(style='sci', scilimits=(1, 4))
        # g.set(xlim=(min_val, max_val), ylim=(min_val_y, max_val_y))
        plt.savefig("test.png")
        plt.show()
