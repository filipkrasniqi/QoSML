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
from joblib import load

import sys
sys.path.insert(0, '../libs/')
sys.path.insert(0, '../pytorch/')
from columns import *

import os
from os.path import join, expanduser

from ns3_dataset import NS3Dataset, Scenario
from routenet_dataset import RoutenetDataset
from understanding_dataset import UnderstandingDataset

plt.show(block=True)
plt.interactive(False)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
arguments = sys.argv
cache_dir = str(arguments[1])
test_less_intensities = False
model_dir = str(arguments[2])
scenario = int(arguments[3])
assert scenario >= 1 and scenario <= 3, "Wrong value describing the scenario"
scenario = [Scenario.LEVEL_1, Scenario.LEVEL_2, Scenario.LEVEL_3][scenario-1]
only_test = True

topology = "abilene"
dataset_id = "ns3"
identifier = "simulation_v1_L{}".format(scenario+1)
if "rf" in model_dir:
    model_name = "random_forest.joblib"
else:
    model_name = "NN.model"

if "L{}".format(scenario+1) in model_dir:
    print("INFO: model is coherent with scenario")
else:
    raise ValueError("ERROR: model not coherent with scenario")

use_ns3, only_low, use_routenet, use_understanding = True, False, False, False
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

dir_log_output = join(
    *[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', 'exported', 'crossvalidation', "results"])
dir_model_output = join(
    *[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', "exported", "crossvalidation"])
path_output = join(*[dir_log_output, model_dir])
path_boxplot = join(*[path_output, "boxplot.csv"])
path_icsp = join(*[path_output, "icsp.csv"])

only_mean_std, also_capacity = True, scenario != Scenario.LEVEL_1
path_x_test, path_real_values, path_predictions = join(*[path_output, "x_test.npy"]), join(*[path_output, "y_test.npy"]), join(*[path_output, "predictions.npy"])

if isfile(path_x_test) and isfile(path_real_values) and isfile(path_icsp) and isfile(path_predictions):
    X_test, real_values, predictions = np.load(path_x_test, allow_pickle=True), np.load(path_real_values, allow_pickle=True), np.load(path_predictions, allow_pickle=True)
    df_other = pd.read_csv(path_icsp, sep=" ", index_col=False)
else:
    if use_ns3:
        coefficient_delay = 1
        if "ms" in cache_dir:
            coefficient_delay = 1000
            print("INFO: Computing delay in ms...")
        """"""
        dataset_container = NS3Dataset(also_pyg=False, scenario=scenario,
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

    if use_ns3:
        if only_mean_std:
            # among the extracted features, only select mean and std (i.e., don't consider quantiles)
            print("WARNING: only considering mean and std as extracted features")
            if also_capacity:
                input_columns = [col for col in dataset_container.dfs_e2e_test.columns if ("traffic" in col or "capacity" in col)]
            else:
                input_columns = [col for col in dataset_container.dfs_e2e_test.columns if ("traffic" in col)]
        else:
            raise EnvironmentError("Never happens")

        if any("capacity" in col for col in input_columns):
            print("INFO: using also capacities in input!")
        else:
            print("WARNING: not using capacity in input!")

    if (not os.path.isdir(dir_log_output)):
        os.mkdir(dir_log_output)

    if (not os.path.isdir(path_output)):
        os.mkdir(path_output)

    if (not os.path.isdir(dir_model_output)):
        raise ValueError("Model directory does not exist")

    # import dataset
    output_columns = [col for col in dataset_container.dfs_e2e_test.columns if "delay_e2e_" in col and "mean" in col]
    need_to_filter = scenario == Scenario.LEVEL_1
    if need_to_filter:
        filter_cols = [col for idx, col in enumerate(output_columns) if idx % dataset_container.num_nodes != int(idx/dataset_container.num_nodes)]
    else:
        filter_cols = output_columns
    df_routing = dataset_container.get_routing_df()
    X_test, real_values = dataset_container.dfs_e2e_test.loc[:, input_columns].to_numpy(), dataset_container.dfs_e2e_test.loc[:, output_columns+["intensity", "simulation"]].to_numpy()
    np.save(path_x_test, X_test)
    np.save(path_real_values, real_values)
    df_other = dataset_container.get_test_iscp_dataframe()
    df_other.to_csv(path_or_buf=path_icsp, sep=' ', index=False, header=True)
    del dataset_container

    try:
        env_name = "{}_{}".format(model_dir, dataset_id)
        env_path = join(*[dir_log_output, env_name])
        model_path = join(*[dir_model_output, env_name, model_name])
        model = load(open(model_path, 'rb'))
    except:
        env_name = "{}".format(model_dir)
        env_path = join(*[dir_log_output, env_name])
        model_path = join(*[dir_model_output, env_name, model_name])
        model = load(open(model_path, 'rb'))

    predictions = model.predict(X_test) * 0.001
    np.save(path_predictions, predictions)
    del model

def columns_od(name, remove_self_loops = True):
    if remove_self_loops:
        return ["{}_{}_{}".format(name, int(i/num_nodes), i%num_nodes) for i in range(num_nodes ** 2) if i%num_nodes != int(i / num_nodes)]
    else:
        return ["{}_{}_{}".format(name, int(i / num_nodes), i % num_nodes) for i in range(num_nodes ** 2)]

real_values = real_values[:, 0:-2]
errors = np.abs(predictions - real_values)
num_nodes = 12
columns_error, columns_delay, columns_predicted = columns_od("abs_error", remove_self_loops = True), columns_od("target", remove_self_loops = True), columns_od("predicted", remove_self_loops = True)
df_error = pd.DataFrame(columns=columns_error, data = {columns_error[i]: errors[:, i] for i in range(num_nodes ** 2 - num_nodes)})
df_delay = pd.DataFrame(columns=columns_delay, data = {columns_delay[i]: real_values[:, i] for i in range(num_nodes ** 2 - num_nodes)})
df_predicted = pd.DataFrame(columns=columns_predicted, data = {columns_predicted[i]: predictions[:, i] for i in range(num_nodes ** 2 - num_nodes)})

# regarding other parameters: intensity, simulation, capacity, ... -> I have the function that does that for me
df = pd.concat([df_error, df_delay, df_predicted, df_other], axis=1, sort=False)
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

simulations_to_show, intensities_to_show, capacities_to_show, pds_to_show = test_simulations, test_intensities,\
                                                                            test_capacities, test_pdelays
df_filtered_sim = filter_df_sims(df, simulations_to_show)

sicp = list(itertools.product(simulations_to_show, intensities_to_show,
                                                            capacities_to_show, pds_to_show))
total_length = len(sicp)

box_plot = True
print("INFO: starting to compute RMSE")
if box_plot and not isfile(path_boxplot):
    rmse_values_intensity, stds_values_intensity, simulations, intensities, capacities, propagation_delays = [], [], [], [], [], []
    for idx_sicp, (c_sim, c_int, c_cap, c_pd) in enumerate(sicp):
        c_sim, c_int, c_cap, c_pd = int(c_sim), int(c_int), int(c_cap), int(c_pd)
        current_df = filter_df(df, c_sim, c_int, c_cap, c_pd)
        current_targets, current_predictions = current_df.loc[:, [col for col in current_df.columns if "target" in col]], \
                                               current_df.loc[:, [col for col in current_df.columns if "predic" in col]]
        rmse = np.power(mean_squared_error(current_targets.values.flatten(), current_predictions.values.flatten()), 0.5)
        std = np.std(current_predictions.values.flatten())
        rmse_values_intensity.append(rmse)
        simulations.append(c_sim)
        intensities.append(c_int)
        capacities.append(c_cap)
        propagation_delays.append(c_pd)
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
    boxplot_current_model = pd.DataFrame(data={"RMSE": rmse_values_intensity + stds_values_intensity, "Simulation": np.concatenate((simulations, simulations)), "Intensity": np.concatenate((intensities, intensities)), "Capacity": np.concatenate((capacities, capacities)), "P.Delay": np.concatenate((propagation_delays, propagation_delays)), "is_std": np.concatenate((np.repeat(1, len(intensities)), np.repeat(0, len(intensities))))})
    boxplot_current_model.to_csv(path_or_buf=path_boxplot, sep=' ', index=False, header=False)

scatter_plot = True
selected_columns = ["_0_9", "_10_1", "_3_7"]

def filter_df_col(df, cols):
    columns_to_filter = []
    for filter in cols:
        columns_current_filter = [col for col in df.columns if col.endswith(filter)]
        columns_to_filter += columns_current_filter
    columns_to_filter += list(df_other.columns.values)
    return df.loc[:, columns_to_filter]

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