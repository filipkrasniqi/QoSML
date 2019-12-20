import pandas as pd
pd.set_option('display.max_columns', None)#or define a numer
pd.set_option('precision', 9)
import numpy as np

from os import listdir
from os.path import join, expanduser

import seaborn as sns

import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

from scipy import stats
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(12,8.5)})
sns.set(font_scale=1.7)

# matplotlib.use('TkAgg')

import sys
sys.path.insert(0, '../libs/')
from columns import *
from sklearn import feature_selection
import torch
from ns3_dataset import NS3Dataset, Scenario
import itertools

arguments = ["rf.py", "cache_v2", "ns3", "rf_test", "abilene", "v1_L1", "all", True, 1]

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
intensities_to_show = [0,4,9]
num_sims_to_show = 1
sims_to_show = list(range(0, 10))# [156]# list(range(num_sims_to_show))
# chosen_intensities = all_intensities[list(range(0, min(len(all_intensities), num_intensities_to_show)))]
chosen_intensities = ["intensity_{}_{}".format(i, s) for i, s in itertools.product(intensities_to_show, sims_to_show)]
window_size = 50

dfs_e2e_test, df_tm, df_dropped = dataset_container.get_dataframe_visualization(test_environments = chosen_environments, test_int_sim = chosen_intensities, window_size = window_size)
df_intensity_sim = dfs_e2e_test["intensity_simulation"].str.split("_", n = 2, expand = True).loc[:,[1,2]]
df_intensity_sim.columns = ["intensity", "simulation"]
dfs_e2e_test["intensity"] = "intensity_" + df_intensity_sim.loc[:,"intensity"]
dfs_e2e_test["simulation"] = "simulation_" + df_intensity_sim.loc[:,"simulation"]
df_intensity_sim = df_tm["intensity_simulation"].str.split("_", n = 2, expand = True).loc[:,[1,2]]
df_intensity_sim.columns = ["intensity", "simulation"]
df_tm["intensity"] = "intensity_" + df_intensity_sim.loc[:,"intensity"]
df_dropped["intensity"] = "intensity_" + df_intensity_sim.loc[:,"intensity"]
df_dropped["simulation"] = "simulation_" + df_intensity_sim.loc[:,"simulation"]

print("Visualization {}, TM: {}".format(dfs_e2e_test.shape, df_tm.shape))
log_cum_dist = False
log_traffic_wrt_capacities = False
log_heatmap = False
log_throughput = False
log_dropped = False
log_delay = True
log_capacity_level_2 = False
log_pd_level_3 = False

plot_only_dist = True

do_pca = False

max_num_nodes = dataset_container.max_num_nodes
num_nodes = dataset_container.num_nodes

range_tm = range(30, 31)
range_delay = range(30, 31)
num_simulations_to_show = range(0, 1)
simulations_to_show = [env for i, env in enumerate(df_tm.environment.unique()) if i in num_simulations_to_show]

only_tm_cols = dataset_container.traffic_cols
only_delay_cols = dataset_container.delay_cols
only_capacity_cols = dataset_container.capacity_cols
tm_cols = [col for col in dfs_e2e_test.columns if "traffic" in col and "mean" in col]
e2e_cols = [col for col in dfs_e2e_test.columns if "delay_e2e_" in col and "mean" in col]
load_cols = [col for col in dfs_e2e_test.columns if "load" in col and "mean" in col]
dropped_cols = [col for col in dfs_e2e_test.columns if "dropped" in col and "mean" in col]

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

df_routing = dataset_container.get_routing_df()
df_capacities = dataset_container.get_capacities_df()
df_tm_single_simulation = df_tm.where(df_tm["environment"] == all_environments[0]).dropna().reset_index()
df_tm_single_simulation.head()
if log_traffic_wrt_capacities:
    for i, col in enumerate([col for col in df_tm_single_simulation.columns if "capacity" in col]):
        if df_tm_single_simulation.loc[0, col] > 0:
            capacity_to_check = "_".join(col.split("_")[1:3])
            cols_to_check = [col for col in df_routing.columns if "link_{}".format(capacity_to_check) in col]
            related_OD_flows = [col.split(" ")[0] for col in cols_to_check if df_routing.loc[0, col] == 1]
            traffic_cols = ["traffic_{}".format("_".join(col.split("_")[1:3])) for col in related_OD_flows]
            df_traffic_summed = df_tm_single_simulation[traffic_cols].sum(axis = 1, skipna = True)

            df_tm_single_simulation["capacity_{}".format(capacity_to_check)].head()

            # time series: each value is on one period. Just represent the values in order
            df = pd.DataFrame(data = {"Capacity": df_tm_single_simulation["capacity_{}".format(capacity_to_check)], "Traffic": df_traffic_summed})
            df["period"] = list(range(df.shape[0]))

            plt.figure()
            g = sns.FacetGrid(df, height=8, aspect=1.5)
            first_node = int(capacity_to_check.split("_")[0])
            second_node = int(capacity_to_check.split("_")[1])
            g.fig.suptitle("Link between nodes {} - {}".format(first_node, second_node)) # can also get the figure from plt.gcf()
            g = (g.map(plt.plot, "period", "Capacity", marker=".")).add_legend()
            g = (g.map(plt.plot, "period", "Traffic", marker=".", color="red")).add_legend()
            p_vals = min(df[["Capacity", "Traffic"]].min()), max(df[["Capacity", "Traffic"]].max())
            delta = (p_vals[1] - p_vals[0]) / 10
            g.axes[0,0].set_ylim(p_vals[0] - delta, p_vals[1] + delta)
            plt.savefig("test.png")
            plt.show()

df = dfs_e2e_test[tm_cols+["intensity"]]
df_columns = [column for column in df.columns if "_" in column]

if log_cum_dist:
    plt.title("Traffic: cumulative over the OD flows")
    for index, intensity in enumerate(["intensity_{}".format(intensity.split("_")[1]) for intensity in df.intensity.unique()]):
        current_regime_vals = df.where(df.intensity == intensity).dropna()
        #shift_global_index = index * current_regime_vals.shape[0]
        cumulative_tr = np.zeros(current_regime_vals.shape[0])
        for period in range(0,current_regime_vals.shape[0]):
            for col in df_columns:
                if "traffic" in col:
                    cumulative_tr[period] += current_regime_vals[col].iloc[period]
        sns.distplot(cumulative_tr, color=palette[intensity])
        plt.savefig("cumulative_{}.png".format(intensity))
        plt.show()

if log_heatmap:
    df_heatmap = dfs_e2e_test[tm_cols+e2e_cols+["intensity"]].sample(frac=0.1, replace=False, random_state=1)
    print("Size: {}".format(df_heatmap.shape))
    input_columns = tm_cols
    output_columns = e2e_cols

    mutual_info_matrix = []

    for i, input_col in enumerate(input_columns):
        mutual_info_matrix.append([])
        for o, output_col in enumerate(output_columns):
            features = df_heatmap[input_col]
            targets = df_heatmap[output_col]
            mutual_info = feature_selection.mutual_info_regression(features.values.reshape(-1,1), targets.values.reshape(-1,1))

            mutual_info_matrix[i].append(mutual_info)

    mutual_info_matrix_copy = []
    for (i, m_i_m) in enumerate(mutual_info_matrix):
        mutual_info_matrix_copy.append([])
        for (j, m_i) in enumerate(m_i_m):
            mutual_info_matrix_copy[i].append(m_i[0])
    sns.heatmap(mutual_info_matrix_copy)
    plt.savefig("heatmap.png")
    plt.show()

simulations = dfs_e2e_test.simulation.unique()
select_sim = simulations[min(len(simulations)-1, 2)]
for select_sim in simulations:
    if log_throughput:
        df_load = dfs_e2e_test[load_cols+["intensity"]+["simulation"]]
        df_load.columns = ["throughput_{}_{}".format(col.split("_")[-2],col.split("_")[-1]) for col in load_cols] + ["intensity", "simulation"]
        # filter by simulation
        df_load = df_load.where(select_sim == df_load.simulation).dropna()
        # select the bottleneck link automatically, i.e., link that according to the routing is used in more
        OD_pairs = list(itertools.product(list(range(dataset_container.max_num_nodes)), list(range(dataset_container.max_num_nodes))))

        count_edges = np.array([0 for _ in OD_pairs])
        for o, d in OD_pairs:
            idx_group_cols = o * dataset_container.max_num_nodes + d
            offset = idx_group_cols * (dataset_container.max_num_nodes ** 2)
            cols = df_routing.columns[offset:offset+dataset_container.max_num_nodes**2]
            # cols = df_routing.columns[idxs_cols] # [col for col in df_routing.columns if "OD_{}_{} ".format(o, d) in col]
            instances = np.array(df_routing[cols].values).squeeze(0)
            count_edges += instances
        bottleneck_link_pair = OD_pairs[np.argmax(count_edges)]
        first_node = bottleneck_link_pair[0]
        second_node = bottleneck_link_pair[1]

        capacity_to_check = "capacity_{}_{}".format(bottleneck_link_pair[0], bottleneck_link_pair[1])
        load_to_check = "throughput_{}_{}".format(bottleneck_link_pair[0], bottleneck_link_pair[1])
        capacity_vals = df_capacities.loc[0,capacity_to_check]
        throughput_vals = df_load[load_to_check]

        # time series: each value is on one period. Just represent the values in order
        df = pd.DataFrame(data={"Capacity": np.array([capacity_vals]).repeat(len(throughput_vals)),
                                "Traffic": throughput_vals,
                                "Intensity": df_load.intensity})
        df["period"] = list(range(df.shape[0]))

        # visualize throughput on singularly to show how throughput evolves
        if not plot_only_dist:
            plt.figure()
            g = sns.FacetGrid(df, height=8, aspect=1.5)
            g.fig.suptitle(
                "Link between nodes {} - {}".format(first_node, second_node))  # can also get the figure from plt.gcf()
            g = (g.map(plt.plot, "period", "Capacity", marker=".")).add_legend()
            g = (g.map(plt.plot, "period", "Traffic", marker=".", color="red")).add_legend()
            p_vals = min(df[["Capacity", "Traffic"]].min()), max(df[["Capacity", "Traffic"]].max())
            delta = (p_vals[1] - p_vals[0]) / 10
            g.axes[0, 0].set_ylim(p_vals[0] - delta, p_vals[1] + delta)
            plt.savefig("test.png")
            plt.show()

            for intensity in df_load.intensity.unique():
                capacity_vals = df_capacities.loc[0, capacity_to_check]
                throughput_vals = df_load.where(df_load.intensity == intensity).dropna()[load_to_check]

                # time series: each value is on one period. Just represent the values in order
                df = pd.DataFrame(data={"Capacity": np.array([capacity_vals]).repeat(len(throughput_vals)),
                                        "Traffic": throughput_vals})
                df["period"] = list(range(df.shape[0]))

                plt.figure()
                g = sns.FacetGrid(df, height=8, aspect=1.5)
                first_node = bottleneck_link_pair[0]
                second_node = bottleneck_link_pair[1]
                g.fig.suptitle(
                    "Link between nodes {} - {}, simulation {}".format(first_node, second_node, select_sim))  # can also get the figure from plt.gcf()
                # g = (g.map(plt.plot, "period", "Capacity", marker=".")).add_legend()
                g = (g.map(plt.plot, "period", "Traffic", marker=".", color="red")).add_legend()
                p_vals = min(df[["Traffic"]].min()), max(df[["Traffic"]].max())
                delta = (p_vals[1] - p_vals[0]) / 10
                g.axes[0, 0].set_ylim(p_vals[0] - delta, p_vals[1] + delta)
                plt.savefig("throughput_{}.png".format(intensity))
                plt.show()


        # same but distribution
        for intensity in df_load.intensity.unique():
            throughput_vals = df_load.where(df_load.intensity == intensity).dropna()[load_to_check]
            throughput_vals = [float(format(val / 10 ** 6, ".8f")) for val in throughput_vals]
            # time series: each value is on one period. Just represent the values in order

            plt.figure()
            first_node = bottleneck_link_pair[0]
            second_node = bottleneck_link_pair[1]

            g = sns.distplot(throughput_vals, color=palette[intensity])
            g.set_yticklabels(labels=[])
            # g.set_xticklabels(labels=g.get_xticklabels(), rotation=30, ha='right')
            plt.xlabel('Throughput values [Mbps]', fontsize=20)
            plt.savefig("throughput_dist_{}.png".format(intensity))
            plt.show()

if log_dropped:
    # df_dropped = dfs_e2e_test[dropped_cols + ["intensity"] + ["simulation"]]
    simulations = df_dropped.simulation.unique()
    select_sim = simulations[min(len(simulations) - 1, 0)]
    # filter by simulation
    #df_dropped = df_dropped.where(select_sim == df_dropped.simulation).dropna()
    # select the bottleneck link automatically, i.e., link that according to the routing is used in more
    OD_pairs = list(itertools.product(list(range(dataset_container.max_num_nodes)),
                                      list(range(dataset_container.max_num_nodes))))

    count_edges = np.array([0 for _ in OD_pairs])
    for o, d in OD_pairs:
        idx_group_cols = o * dataset_container.max_num_nodes + d
        offset = idx_group_cols * (dataset_container.max_num_nodes ** 2)
        cols = df_routing.columns[offset:offset + dataset_container.max_num_nodes ** 2]
        # cols = df_routing.columns[idxs_cols] # [col for col in df_routing.columns if "OD_{}_{} ".format(o, d) in col]
        instances = np.array(df_routing[cols].values).squeeze(0)
        count_edges += instances
    bottleneck_link_pair = OD_pairs[np.argmax(count_edges)]

    capacity_to_check = "capacity_{}_{}".format(bottleneck_link_pair[0], bottleneck_link_pair[1])
    load_to_check = "dropped_{}_{}".format(bottleneck_link_pair[0], bottleneck_link_pair[1])
    capacity_vals = df_capacities.loc[0, capacity_to_check]
    dropped_vals = df_dropped[load_to_check]

    # visualize dropped singularly to show how it evolves
    if not plot_only_dist:
        for intensity in df_dropped.intensity.unique():
            capacity_vals = df_capacities.loc[0, capacity_to_check]
            dropped_vals = df_dropped.where(df_dropped.intensity == intensity).dropna()[load_to_check]

            # time series: each value is on one period. Just represent the values in order
            df = pd.DataFrame(data={"Capacity": np.array([capacity_vals]).repeat(len(dropped_vals)),
                                    "Dropped": dropped_vals})
            df["period"] = list(range(df.shape[0]))

            plt.figure()
            g = sns.FacetGrid(df, height=8, aspect=1.5)
            first_node = bottleneck_link_pair[0]
            second_node = bottleneck_link_pair[1]
            g.fig.suptitle(
                "Link between nodes {} - {}".format(first_node,
                                                    second_node))
            g = (g.map(plt.plot, "period", "Dropped", marker=".", color="red")).add_legend()
            p_vals = min(df[["Dropped"]].min()), max(df[["Dropped"]].max())
            delta = (p_vals[1] - p_vals[0]) / 10
            g.axes[0, 0].set_ylim(p_vals[0] - delta, p_vals[1] + delta)
            plt.savefig("dropped_{}.png".format(intensity))
            plt.show()

    # visualize throughput on singularly to show how throughput evolves
    # dropped_vals = df_dropped[load_to_check]
    for intensity, simulation in itertools.product(df_dropped.intensity.unique(), df_dropped.simulation.unique()):
        current_df_dropped = df_dropped.where(df_dropped.intensity == intensity).dropna()
        current_df_dropped = current_df_dropped.where(current_df_dropped.simulation == simulation).dropna()[load_to_check]
        dropped_vals = current_df_dropped#.where(current_df_dropped > 0).dropna()
        if np.sum(dropped_vals) <= 0:
            print("WARNING: No drops!")
        else:
            if "0" in intensity or "4" in intensity:
                print("AH")
            # time series: each value is on one period. Just represent the values in order

            plt.figure()
            first_node = bottleneck_link_pair[0]
            second_node = bottleneck_link_pair[1]

            g = sns.distplot(dropped_vals, color=palette[intensity])
            g.set_yticklabels(labels=[])
            plt.xlabel('N. dropped packets in link between nodes {}-{}'.format(first_node, second_node), fontsize=20)


            plt.savefig("dropped_dist_{}.png".format(intensity))
            plt.show()

if log_delay:
    df_delay = dfs_e2e_test[e2e_cols + ["intensity"] + ["simulation"]]
    simulations = dfs_e2e_test.simulation.unique()
    select_sim = simulations[min(len(simulations) - 1, 2)]
    # filter by simulation
    df_delay = df_delay.where(select_sim == df_delay.simulation).dropna()
    # select the bottleneck link automatically, i.e., link that according to the routing is used in more
    OD_pairs = list(itertools.product(list(range(dataset_container.max_num_nodes)),
                                      list(range(dataset_container.max_num_nodes))))

    current_longest_path = 0
    od_longest = (0, 0)
    for o, d in OD_pairs:
        idx_group_cols = o * dataset_container.max_num_nodes + d
        offset = idx_group_cols * (dataset_container.max_num_nodes ** 2)
        cols = df_routing.columns[offset:offset + dataset_container.max_num_nodes ** 2]
        # cols = df_routing.columns[idxs_cols] # [col for col in df_routing.columns if "OD_{}_{} ".format(o, d) in col]
        traversed_links = np.sum(df_routing[cols].values).squeeze(0)
        if traversed_links > current_longest_path:
            current_longest_path = traversed_links
            od_longest = (o, d)

    load_to_check = "mean_delay_e2e_{}_{}".format(od_longest[0], od_longest[1])
    delay_vals = df_delay[load_to_check]

    # time series: each value is on one period. Just represent the values in order
    df = pd.DataFrame(data={"Delay": delay_vals,
                            "Intensity": df_delay.intensity})
    df["period"] = list(range(df.shape[0]))

    if not plot_only_dist:
        plt.figure()
        g = sns.FacetGrid(df, height=8, aspect=1.5)
        first_node = od_longest[0]
        second_node = od_longest[1]
        g.fig.suptitle(
            "Link between nodes {} - {}".format(first_node, second_node))  # can also get the figure from plt.gcf()
        g = (g.map(plt.plot, "period", "Delay", marker=".", color="red")).add_legend()
        p_vals = min(df[[ "Delay"]].min()), max(df[[ "Delay"]].max())
        delta = (p_vals[1] - p_vals[0]) / 10
        g.axes[0, 0].set_ylim(p_vals[0] - delta, p_vals[1] + delta)
        plt.savefig("delay.png")
        plt.show()

        # visualize delay on singularly to show how it evolves
        for intensity in df_delay.intensity.unique():
            delay_vals = df_delay.where(df_delay.intensity == intensity).dropna()[load_to_check]
            throughput_vals = [float(format(val, ".8f")) for val in delay_vals]
            # time series: each value is on one period. Just represent the values in order
            df = pd.DataFrame(data={
                                    "Delay": delay_vals})
            df["period"] = list(range(df.shape[0]))

            plt.figure()
            g = sns.FacetGrid(df, height=8, aspect=1.5)
            first_node = od_longest[0]
            second_node = od_longest[1]
            g.fig.suptitle(
                "Link between nodes {} - {}".format(first_node,
                                                    second_node))  # can also get the figure from plt.gcf()
            # g = (g.map(plt.plot, "period", "Capacity", marker=".")).add_legend()
            g = (g.map(plt.plot, "period", "Delay", marker=".", color="red")).add_legend()
            p_vals = min(df[["Delay"]].min()), max(df[["Delay"]].max())
            delta = (p_vals[1] - p_vals[0]) / 10
            g.axes[0, 0].set_ylim(p_vals[0] - delta, p_vals[1] + delta)
            plt.savefig("delay_{}.png".format(intensity))
            plt.show()

    # same but distribution
    for intensity in df_delay.intensity.unique():
        delay_vals = df_delay.where(df_delay.intensity == intensity).dropna()[load_to_check]
        # time series: each value is on one period. Just represent the values in order

        plt.figure()
        first_node = od_longest[0]
        second_node = od_longest[1]

        g = sns.distplot(delay_vals, color=palette[intensity])
        g.set_yticklabels(labels=[])
        plt.xlabel('End-to-end delay values for OD {}-{} [Mbps]'.format(first_node, second_node), fontsize=20)
        plt.savefig("delay_dist_{}.png".format(intensity))
        plt.show()

# show in that ...
idxs_to_show = [0, 4, 6]
if log_capacity_level_2:
    capacities_dict = dataset_container.get_capacities(idxs_to_show)
    capacities_df, links_df = np.array([]), np.array([])
    for link_to_show in capacities_dict.keys():
        current = capacities_dict[link_to_show]
        capacities_df = np.append(capacities_df, current)
        links_df = np.append(links_df, np.repeat(link_to_show, len(current)))

    df = pd.DataFrame(data={"Capacity [Mbps]": capacities_df, "Link": links_df})
    plt.figure()
    g = sns.FacetGrid(df, height=8, aspect=1.5, col="Link")
    g = g.map(sns.distplot, "Capacity [Mbps]")
    plt.savefig("test.png")
    plt.show()


if log_pd_level_3:
    pdelays_dict = dataset_container.get_pds(idxs_to_show)

    pdelays_df, links_df = np.array([]), np.array([])
    for link_to_show in pdelays_dict.keys():
        current = pdelays_dict[link_to_show]
        pdelays_df = np.append(pdelays_df, current)
        links_df = np.append(links_df, np.repeat(link_to_show, len(current)))

    df = pd.DataFrame(data={"Propagation Delay [ms]": pdelays_df, "Link": links_df})
    plt.figure()
    g = sns.FacetGrid(df, height=8, aspect=1.5, col="Link")
    g = g.map(sns.distplot, "Propagation Delay [ms]")
    plt.savefig("test.png")
    plt.show()

    all_pds = dataset_container.get_pds_union
    plt.figure()
    g = sns.FacetGrid(pd.DataFrame(data={"Propagation Delay [ms]": pdelays_df}), height=8, aspect=1.5)
    g = g.map(sns.distplot, "Propagation Delay [ms]")
    plt.savefig("test.png")
    plt.show()

if do_pca:
    filtered_tm_cols = [col for i, col in enumerate(tm_cols) if int(i/max_num_nodes) != i%max_num_nodes]
    filtered_e2e_cols = [col for i, col in enumerate(e2e_cols) if int(i/max_num_nodes) != i%max_num_nodes]
    sampled_df = dfs_e2e_test.sample(frac=0.1, replace=False, random_state=1)
    df_heatmap_tm = sampled_df[filtered_tm_cols]
    df_heatmap_e2e = sampled_df[filtered_e2e_cols]
    # sns.heatmap(df_heatmap.corr())
    plt.savefig("test.png")
    plt.show()

    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=32, batch_size=512)
    df_after_pca = ipca.fit_transform(df_heatmap_tm)
    df_after_pca = df_after_pca.reshape(df_after_pca.shape[1], -1)

    sns.heatmap(np.corrcoef(df_after_pca))
    plt.savefig("test.png")
    plt.show()

    import scipy
    pearsons_first = []
    for tm_col in filtered_tm_cols:
        for e2e_col in filtered_e2e_cols:
            pearsons_first.append(scipy.stats.pearsonr(df_heatmap_tm.loc[:, tm_col].values, df_heatmap_e2e.loc[:, e2e_col].values)[0])

    pearsons_first = np.array(pearsons_first).reshape(len(filtered_tm_cols), len(filtered_e2e_cols))

    pearsons_second = []
    for hidden in df_after_pca:
        for e2e_col in filtered_e2e_cols:
            pearsons_second.append(scipy.stats.pearsonr(hidden, df_heatmap_e2e.loc[:, e2e_col].values)[0])

    pearsons_second = np.array(pearsons_second).reshape(df_after_pca.shape[0], len(filtered_e2e_cols))

    print("OK")