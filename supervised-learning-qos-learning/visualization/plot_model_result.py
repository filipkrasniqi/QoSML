import itertools

import os
from os.path import join, expanduser, isfile

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

plt.show(block=True)
plt.interactive(False)

palette = palette_NN = {
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

palette_RF = {
    "intensity_0":"#535cfc",
    "intensity_1":"#3e48fc",
    "intensity_2":"#2834fb",
    "intensity_3":"#131ffb",
    "intensity_4":"#0411f4",
    "intensity_5":"#040fde",
    "intensity_6":"#030ec9",
    "intensity_7":"#030cb3",
    "intensity_8":"#030b9e",
    "intensity_9":"#020988",
}

#["_0_9", "_10_1", "_3_7"]
OD_flows = ["3-7", "0-9", "10-1"]
markers_distinct_OD = ["^", "x", "s"]

models = ["NN", "RF"]

palette_per_model = [
    palette_NN,
    palette_RF
]

markers_distinct_models = ["x", "s"]

palette_intensity_val = {
    idx+1: palette["intensity_{}".format(idx)] for idx, _ in enumerate(palette)
}

markers_intensity_OD = [
    markers_distinct_OD[idx_OD] for key, (idx_OD, OD) in itertools.product(palette_intensity_val.keys(), enumerate(OD_flows))
]

"""
markers_intensity_model = [
    markers_distinct_models[idx_model] for key, (idx_model, model) in itertools.product(palette_intensity_val.keys(), enumerate(models))
]
"""

palette_intensity_OD = {
    "{}_{}".format(key, OD): palette_intensity_val[key] for key, OD in itertools.product(palette_intensity_val.keys(), OD_flows)
}

palette_intensity_model = {
    "{}_{}".format(key, model): palette_per_model[idx_model][list(palette.keys())[idx_key]] for (idx_key, key), (idx_model, model) in itertools.product(enumerate(palette_intensity_val.keys()), enumerate(models))
}

dir_log_output = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', 'exported', 'crossvalidation', "results"])
path_boxplot, path_scatterplot = join(*[dir_log_output, "boxplot"]), join(*[dir_log_output, "scatterplot"])
boxplot = pd.DataFrame()

do_boxplot = False
if do_boxplot:

    for boxplot_result in os.listdir(path_boxplot):
        vals_current_model = boxplot_result.split("_")  # they all follow pattern: <model_name>_search_vX_L<Scenario>
        model_name = vals_current_model[0].upper()
        path_file = join(*[path_boxplot, boxplot_result, "boxplot.csv"])
        if isfile(path_file) and "R" in model_name:
            print("AH")
        if not isfile(path_file):
            path_file = join(*[path_boxplot, boxplot_result, "boxplot_test_v2.csv"])
        names = ["RMSE", "Intensity", "RMSE/STD"]
        names_v2 = ["RMSE", "Simulation", "Intensity", "Capacity", "P.Delay", "RMSE/STD"]
        try:
            boxplot_current_model = pd.read_csv(path_file, sep=" ", names=names, index_col=False)
        except:
            boxplot_current_model = pd.read_csv(path_file, sep=" ", names=names_v2,
                                                index_col=False)
            boxplot_current_model = boxplot_current_model.loc[:, names]

        boxplot_current_model = boxplot_current_model.where(boxplot_current_model["RMSE/STD"] == 1).dropna()
        boxplot_current_model["Model"] = model_name
        boxplot_current_model["Intensity"] = boxplot_current_model["Intensity"].apply(lambda x: int(x+1))
        boxplot_current_model["Scenario"] = "Scenario {}".format(vals_current_model[3].split("L")[1])
        intensity_model = ["intensity_{}_{}".format(intensity, model_name) for intensity in boxplot_current_model.Intensity]
        boxplot_current_model["Intensity, Model"] = intensity_model

        boxplot = pd.concat([boxplot, boxplot_current_model])

    g = sns.catplot(data=boxplot, x="Model", y="RMSE", hue="Intensity", col="Scenario", kind="boxen", palette=palette_intensity_val, legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("test.png")
    plt.show(block=True)
    g.despine(left=True)

scatterplot = pd.DataFrame()
do_scatterplot = True
if do_scatterplot:
    for model_for_scatterplot in os.listdir(path_scatterplot):
        vals_current_model = model_for_scatterplot.split("_")  # they all follow pattern: <model_name>_search_vX_L<Scenario>
        model_name = vals_current_model[0].upper()
        path_model = join(*[path_scatterplot, model_for_scatterplot])
        current_scenario = int(vals_current_model[3].split("L")[1])
        scenarios_fraction_sample = [0.7, 0.7, 0.5]
        for scatterplot_result in [scatter for scatter in os.listdir(path_model) if "scatter" in scatter]:
            column = "-".join(scatterplot_result.split(".csv")[0].split("scatter_")[1].split("_"))
            if column in OD_flows:
                path_file = join(*[path_model, scatterplot_result])
                scatterplot_current_model = pd.read_csv(path_file, sep=" ", index_col=False).sample(frac=scenarios_fraction_sample[current_scenario-1])

                scatterplot_current_model["Model"] = model_name
                scatterplot_current_model["OD"] = column
                scatterplot_current_model["Intensity"] = scatterplot_current_model["Intensity"].apply(lambda x: int(x + 1))
                scatterplot_current_model["Scenario"] = current_scenario
                scatterplot_current_model["Intensity, OD"] = scatterplot_current_model.apply(lambda x: "{}_{}".format(x["Intensity"], x["OD"]), axis=1)
                scatterplot_current_model["Intensity, Model"] = scatterplot_current_model.apply(
                    lambda x: "{}_{}".format(x["Intensity"], x["Model"]), axis=1)
                scatterplot = pd.concat([scatterplot, scatterplot_current_model])

    markers_intensity_model = []
    for intensity_model in scatterplot["Intensity, Model"].unique():
        if "NN" in intensity_model:
            markers_intensity_model.append("x")
        elif "RF" in intensity_model:
            markers_intensity_model.append("s")
        else:
            print("MERDA")
    g = sns.lmplot(data=scatterplot, x="Avg.Targets", y="Avg.Abs.Errors", col="Scenario", row="OD",
                   hue="Intensity, Model", palette=palette_intensity_model, markers=markers_intensity_model, legend=False,
                   scatter=True, fit_reg=False)
    # hue intensity-OD: # g = sns.lmplot(data=scatterplot, x="Avg.Targets", y="Avg.Abs.Errors", col="Scenario", row="Model", hue="Intensity, OD", palette=palette_intensity_OD, markers=markers_intensity_OD, legend=False, scatter=True, fit_reg=False)
    min_x, max_x, min_y, max_y = min(scatterplot["Avg.Targets"].values), max(scatterplot["Avg.Targets"].values), min(scatterplot["Avg.Abs.Errors"].values), max(scatterplot["Avg.Abs.Errors"].values)
    delta_x, delta_y = max_x - min_x, max_y - min_y
    min_x, max_x, min_y, max_y = min_x - delta_x * 0.1, max_x + delta_x * 0.1, min_y - delta_y * 0.1, max_y + delta_y * 0.1
    g.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("test.png")
    plt.show(block=True)
    g.despine(left=True)

    """
    g = sns.palplot(sns.color_palette("Blues_d"))
    plt.savefig("test.png")
    plt.show(block=True)

    g = sns.palplot(sns.color_palette("Reds_d"))
    plt.savefig("test.png")
    plt.show(block=True)
    """