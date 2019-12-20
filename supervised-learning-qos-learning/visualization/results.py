import pandas as pd
pd.set_option('display.max_columns', None)#or define a numer
pd.set_option('precision', 9)
import numpy as np

from os import listdir
from os.path import join, expanduser

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0)
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(12,8.5)})
sns.set(font_scale=1.5)

# matplotlib.use('TkAgg')

import sys
sys.path.insert(0, '../libs/')
from columns import *
from sklearn import feature_selection
import torch
from ns3_dataset import NS3Dataset, Scenario
import itertools
from joblib import load
import os

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# rf_no_pca_v1, rf_L2v1, rf_test
#L3:
# arguments = ["rf.py", "cache_v2", "ns3", "rf_search_v1_L3", "abilene", "v1_L3", "all", False, 3]
#L2:
# arguments = ["rf.py", "cache_v2", "ns3", "rf_search_v1_L2", "abilene", "v1_L2", "all", False, 2]
#L1:
arguments = ["rf.py", "cache_v2", "ns3", "rf_search_v1_L1", "abilene", "v1_L1", "all", False, 1]


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
    test_dataset = dataset_container = NS3Dataset(only_test=only_test, scenario=scenario, generate_tensors=False,
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

all_environments, all_intensities = dataset_container.get_unique_simulation_intensities(test=only_test)
num_environments_to_show = 1
chosen_environments = all_environments[list(range(0, min(len(all_environments), num_environments_to_show)))]
num_intensities_to_show = 9
chosen_intensities = all_intensities[list(range(0, min(len(all_intensities), num_intensities_to_show)))]

df_test = dataset_container.dfs_e2e_test
max_num_nodes = dataset_container.max_num_nodes
num_nodes = dataset_container.num_nodes

range_tm = range(30, 31)
range_delay = range(30, 31)
num_simulations_to_show = range(0, 1)


only_tm_cols = dataset_container.traffic_cols
only_delay_cols = dataset_container.delay_cols
only_capacity_cols = dataset_container.capacity_cols
tm_cols = [col for col in df_test.columns if "traffic" in col and "mean" in col]
e2e_cols = [col for col in df_test.columns if "delay_e2e_" in col and "mean" in col]
load_cols = [col for col in df_test.columns if "load" in col and "mean" in col]
dropped_cols = [col for col in df_test.columns if "dropped" in col and "mean" in col]
# df_train = dataset_container.dfs_e2e_train_vis

filter_columns = ["mean", "std", "capacity"]
prefixes_cols_self_edges = ["{}_{}".format(i, i) for i in range(dataset_container.max_num_nodes)]
prefixes_cols_edges = dataset_container.get_edge_prefixes()
only_mean_std = True
also_capacity = (scenario == Scenario.LEVEL_2 or scenario == Scenario.LEVEL_3)

if use_ns3:
    if only_mean_std:
        # among the extracted features, only select mean and std (i.e., don't consider quantiles)
        print("WARNING: only considering mean and std as extracted features")
        if also_capacity:
            input_columns = [col for col in df_test.columns if ("traffic" in col and ("mean" in col or "std" in col) and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges)) or ("capacity" in col and any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_edges))]
        else:
            input_columns = [col for col in df_test.columns if ("traffic" in col and ("mean" in col or "std" in col) and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges))]
    else:
        if also_capacity:
            input_columns = [col for col in df_test.columns if ("traffic" in col and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges)) or ("capacity" in col and any(prefix == "_".join(col.split("_")[1:3]) for prefix in prefixes_cols_edges))]
        else:
            input_columns = [col for col in df_test.columns if ("traffic" in col and not any(prefix == "_".join(col.split("_")[2:4]) for prefix in prefixes_cols_self_edges))]

    if any("capacity" in col for col in input_columns):
        print("INFO: using also capacities in input!")
    else:
        print("WARNING: not using capacity in input!")
else:
    input_columns = dataset_container.input_columns
output_columns = [col for col in df_test.columns if "delay_e2e_" in col and "mean" in col]
need_to_filter = scenario == Scenario.LEVEL_1
if need_to_filter:
    filter_cols = [col for idx, col in enumerate(output_columns) if idx % dataset_container.num_nodes != int(idx/dataset_container.num_nodes)]
else:
    filter_cols = output_columns
df_routing = dataset_container.get_routing_df()
del dataset_container
X_test, real_values = df_test.loc[:, input_columns], df_test.loc[:, output_columns+["intensity", "simulation"]]
df_intensity_sim = real_values["intensity"].str.split("_", n = 2, expand = True).loc[:,[1,2]]
df_intensity_sim.columns = ["intensity", "simulation"]
df_capacity_pd = real_values["simulation"].str.split("_", n = 2, expand = True).loc[:,[1,2]]
df_capacity_pd.columns = ["capacity", "pdelay"]
real_values["intensity"] = "intensity_" + df_intensity_sim.loc[:,"intensity"]
real_values["simulation"] = "simulation_" + df_intensity_sim.loc[:,"simulation"]
real_values["capacity"] = "capacity_" + df_capacity_pd.loc[:,"capacity"]
real_values["pdelay"] = "pdelay_" + df_capacity_pd.loc[:,"pdelay"]
del df_test

print("Loading model...")
# /home/filip/backup/jeremie/43/to_backup/best_models/rf_ns3_level1/
# /workspace/ns-allinone-3.29/ns-3.29/exported/crossvalidation/
dir_log_output = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', 'exported', 'crossvalidation', "results"])
dir_model_output = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', "exported", "crossvalidation"])

if (not os.path.isdir(dir_log_output)):
    os.mkdir(dir_log_output)

if (not os.path.isdir(dir_model_output)):
    raise ValueError("Model directory does not exist")

env_name = model_dir
model_name = "random_forest.joblib"
pca_name = "ipca.joblib"
env_path = join(*[dir_log_output, env_name])
model_path = join(*[dir_model_output, env_name, model_name])
pca_path = join(*[dir_model_output, env_name, pca_name])

model = load(open(model_path, 'rb'))
try:
    ipca = load(open(pca_path, 'rb'))
    X_test = ipca.transform(X_test)
except:
    print("WARNING: without PCA")

def predict(X):
    predictions = model.predict(X)
    return predictions

predictions = predict(X_test)
del model

filter_cols_idx = [list(real_values.columns).index(col) for col in filter_cols]
for intensity in real_values.intensity.unique():
    filtered_real_values = real_values.where(real_values["intensity"] == intensity).dropna()
    filtered_predictions = predictions[filtered_real_values.index.values] * 0.001
    filtered_real_values.reset_index(inplace=True)

    std_intensity = np.std(filtered_real_values[filter_cols].values)
    rmse_intensity = np.power(mean_squared_error(filtered_predictions[:, filter_cols_idx], filtered_real_values[filter_cols].values), 0.5)

    print(intensity, mean_absolute_error(filtered_real_values[filter_cols].values, filtered_predictions[:, filter_cols_idx]), std_intensity, rmse_intensity / std_intensity, std_intensity / rmse_intensity)

filtered_real_values, filtered_predictions = real_values[filter_cols].values, predictions[:, filter_cols_idx] * 0.001
# real_values[filter_cols].values, predictions[:, filter_cols_idx]
mse_test, mae_test, r2_test = mean_squared_error(filtered_real_values, filtered_predictions), mean_absolute_error(filtered_real_values, filtered_predictions), r2_score(filtered_real_values, filtered_predictions)

print("INFO: MSE = {}".format(mse_test))
print("INFO: MAE = {}".format(mae_test))
print("INFO: STD test = {}".format(np.std(filtered_real_values)))
# print("INFO: R2 = {}".format(r2_test))

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

def scatter_mae_delay(predicted, real_values, filtered_output=output_columns, simulation_range=list(range(0, 1))):
    df = pd.DataFrame(
        columns=["prediction_{}".format(col) for col in filtered_output] + ["target_{}".format(col) for col in
                                                                            filtered_output])
    filtered_simulations = real_values["simulation"].unique()[simulation_range]
    all_simulations = real_values["simulation"].values
    filtered_rows = [idx for idx, sim in enumerate(all_simulations) if
                     sim in filtered_simulations]
    periods = list(range(0, len(filtered_rows)))

    intensities = real_values["intensity"].values[filtered_rows]
    all_simulations = real_values["simulation"].values[filtered_rows]
    capacities = real_values["capacity"].values[filtered_rows]
    pdelays = real_values["pdelay"].values[filtered_rows]
    df["period"] = [period % 350 for period in periods]

    df["intensity"], df["simulation"], df["capacity"], df["pdelay"] = list(intensities), list(capacities), list(pdelays), list(all_simulations)
    predicted = predicted[filtered_rows]
    real_values = real_values.values[filtered_rows]
    for idx_col, col in enumerate(output_columns):
        if col in filtered_output:
            prediction_col = "prediction_{}".format(col)
            target_col = "target_{}".format(col)
            df[prediction_col] = np.array(list(predicted[:, idx_col]))
            df[prediction_col] = df["prediction_{}".format(col)].multiply(0.001)
            df[target_col] = np.array(list(real_values[:, idx_col]))
            df["mae_{}".format(col)] = abs(df[prediction_col] - df[target_col])
            df["mse_{}".format(col)] = (df[prediction_col] - df[target_col]) ** 2

    sim_col_int = []
    for sim, col, intensity, capacity, prop_delay in itertools.product(*[df.simulation.unique(), filtered_output, df.intensity.unique(), df.capacity.unique(), df.pdelay.unique()]):
        sim_col_int.append((sim, col, intensity, capacity, prop_delay))

    for i, s_c_i in enumerate(sim_col_int):
        sim, col, intensity, capacity, pdelay = s_c_i
        delay_col = "target_{}".format(col)
        mse_col = "mae_{}".format(col)
        current_df = df.where(df["simulation"] == sim).dropna()
        total_min_ae, total_max_ae = min(current_df[mse_col].values), max(current_df[mse_col].values)
        current_df = current_df.where(current_df["intensity"] == intensity).dropna()
        current_df = current_df.where(current_df["capacity"] == capacity).dropna()
        current_df = current_df.where(current_df["pdelay"] == pdelay).dropna()
        min_val = min(current_df[delay_col].values)
        max_val = max(current_df[delay_col].values)

        delta = abs(total_max_ae-total_min_ae) / 10
        min_val_y = total_min_ae - delta
        max_val_y = total_max_ae + delta

        g = sns.scatterplot(current_df[delay_col].values, current_df[mse_col].values, color=palette[intensity])  # , kde_kws = kde_kws)
        g.set_xlabel("Delay vs Absolute Error for OD:{}".format(",".join(col.split("_")[3:5])))
        # g.set_yticklabels(labels=[])
        # g.yaxis.set_label_position("right")

        # g.yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
        g.ticklabel_format(axis='x', style='sci', scilimits=(-3, 0))
        g.ticklabel_format(axis='y', style='sci', scilimits=(-1, 0))

        # g.xaxis.ticklabel_format(style='sci', scilimits=(1, 4))
        g.set(xlim=(min_val, max_val), ylim=(min_val_y, max_val_y))
        plt.savefig("test.png")
        plt.show()

    return df

def dist_mae_intensity(predicted, real_values, filtered_output=output_columns, simulation_range=list(range(0, 1))):
    df = pd.DataFrame(
        columns=["prediction_{}".format(col) for col in filtered_output] + ["target_{}".format(col) for col in
                                                                            filtered_output])
    filtered_simulations = real_values["simulation"].unique()[simulation_range]
    all_simulations = real_values["simulation"].values
    filtered_rows = [idx for idx, sim in enumerate(all_simulations) if
                     sim in filtered_simulations]
    periods = list(range(0, len(filtered_rows)))

    intensities = real_values["intensity"].values[filtered_rows]
    all_simulations = real_values["simulation"].values[filtered_rows]
    capacities = real_values["capacity"].values[filtered_rows]
    pdelays = real_values["pdelay"].values[filtered_rows]
    df["period"] = [period % 350 for period in periods]

    df["intensity"], df["simulation"], df["capacity"], df["pdelay"] = list(intensities), list(capacities), list(pdelays), list(all_simulations)
    predicted = predicted[filtered_rows]
    real_values = real_values.values[filtered_rows]
    for idx_col, col in enumerate(output_columns):
        if col in filtered_output:
            prediction_col = "prediction_{}".format(col)
            target_col = "target_{}".format(col)
            df[prediction_col] = np.array(list(predicted[:, idx_col]))
            df[prediction_col] = df["prediction_{}".format(col)].multiply(0.001)
            df[target_col] = np.array(list(real_values[:, idx_col]))
            df["mae_{}".format(col)] = abs(df[prediction_col] - df[target_col])
            df["mse_{}".format(col)] = (df[prediction_col] - df[target_col]) ** 2

    sim_col_int = []
    for sim, col, intensity in itertools.product(*[df.simulation.unique(), filtered_output, df.intensity.unique()]):
        sim_col_int.append((sim, col, intensity))

    for i, s_c_i in enumerate(sim_col_int):
        sim, col, intensity = s_c_i

        current_df = df.where(df["simulation"] == sim).dropna()
        current_df = current_df.where(df["intensity"] == intensity).dropna()
        mse_col = "mae_{}".format(col)

        # g = sns.FacetGrid(current_df, hue="intensity", height=8, aspect=1.5, palette=palette)
        # g = g.map(sns.distplot, mse_col)
        g = sns.distplot(current_df[mse_col], color=palette[intensity])
        g.set_yticklabels(labels=[])
        g.set_xlabel("Absolute Error distribution for OD:{}".format(",".join(col.split("_")[3:5])))
        g.ticklabel_format(axis='x', style='sci', scilimits=(-3, 0))
        # g.ticklabel_format(style="sci", axis="both", scilimits=(0,0))
        # x_labels = ['{:.5e}'.format(float(label)) for label in g.get_xticklabels()]
        # g.set_xticklabels(labels=x_labels)
        plt.savefig("test.png")
        plt.show()
    return df

def predicted_target_intensity(predicted, real_values, filtered_output=output_columns, simulation_range=list(range(0, 1))):
    df = pd.DataFrame(
        columns=["prediction_{}".format(col) for col in filtered_output] + ["target_{}".format(col) for col in
                                                                            filtered_output])
    filtered_simulations = real_values["simulation"].unique()[simulation_range]
    all_simulations = real_values["simulation"].values
    filtered_rows = [idx for idx, sim in enumerate(all_simulations) if
                     sim in filtered_simulations]
    periods = list(range(0, len(filtered_rows)))

    intensities = real_values["intensity"].values[filtered_rows]
    all_simulations = real_values["simulation"].values[filtered_rows]
    capacities = real_values["capacity"].values[filtered_rows]
    pdelays = real_values["pdelay"].values[filtered_rows]
    df["period"] = [period % 350 for period in periods]

    df["intensity"], df["simulation"], df["capacity"], df["pdelay"] = list(intensities), list(capacities), list(pdelays), list(all_simulations)
    predicted = predicted[filtered_rows]
    real_values = real_values.values[filtered_rows]
    for idx_col, col in enumerate(output_columns):
        if col in filtered_output:
            prediction_col = "prediction_{}".format(col)
            target_col = "target_{}".format(col)
            df[prediction_col] = np.array(list(predicted[:, idx_col]))
            df[prediction_col] = df["prediction_{}".format(col)].multiply(0.001)
            df[target_col] = np.array(list(real_values[:, idx_col]))
            df["mae_{}".format(col)] = abs(df[prediction_col] - df[target_col])
            df["mse_{}".format(col)] = (df[prediction_col] - df[target_col]) ** 2

    sim_col_int = []
    for sim, col, intensity, capacity, pdelay in itertools.product(*[df.simulation.unique(), filtered_output, df.intensity.unique(), df.capacity.unique(), df.pdelay.unique()]):
        sim_col_int.append((sim, col, intensity, capacity, pdelay))

    for i, s_c_i in enumerate(sim_col_int):
        sim, col, intensity, capacity, pdelay = s_c_i

        current_df = df.where(df["simulation"] == sim).dropna()
        current_df = current_df.where(df["intensity"] == intensity).dropna()
        current_df = current_df.where(df["capacity"] == capacity).dropna()
        current_df = current_df.where(df["pdelay"] == pdelay).dropna()

        prediction_col = "prediction_{}".format(col)
        target_col = "target_{}".format(col)

        # g = sns.FacetGrid(current_df, hue="intensity", height=8, aspect=1.5, palette=palette)
        # g = g.map(sns.distplot, mse_col)
        g = sns.lineplot(current_df[prediction_col].index, current_df[prediction_col].values, color="blue")
        g = sns.lineplot(current_df[target_col].index, current_df[target_col].values, color=palette[intensity])
        g.set_yticklabels(labels=[])
        g.set_xlabel("Target vs prediction:{}".format(",".join(col.split("_")[3:5])))
        # x_labels = ['{:.5e}'.format(float(label)) for label in g.get_xticklabels()]
        # g.set_xticklabels(labels=x_labels)
        plt.savefig("test.png")
        plt.show()
    return df
# select the bottleneck link automatically, i.e., link that according to the routing is used in more
OD_pairs = list(itertools.product(list(range(max_num_nodes)),
                                  list(range(max_num_nodes))))

current_longest_path = 0
od_longest = (0, 0)
for o, d in OD_pairs:
    idx_group_cols = o * max_num_nodes + d
    offset = idx_group_cols * (max_num_nodes ** 2)
    cols = df_routing.columns[offset:offset + max_num_nodes ** 2]
    # cols = df_routing.columns[idxs_cols] # [col for col in df_routing.columns if "OD_{}_{} ".format(o, d) in col]
    traversed_links = np.sum(df_routing[cols].values).squeeze(0)
    if traversed_links > current_longest_path:
        current_longest_path = traversed_links
        od_longest = (o, d)

ae_to_show = "mean_delay_e2e_{}_{}".format(od_longest[0], od_longest[1])

# plot abs error per intensity
columns_to_show = [ae_to_show]
scatter_mae_delay(predictions, real_values, filtered_output = columns_to_show)
# predicted_target_intensity(predictions, real_values, filtered_output = columns_to_show)
dist_mae_intensity(predictions, real_values, filtered_output = columns_to_show)
# TODO mostrare invece la