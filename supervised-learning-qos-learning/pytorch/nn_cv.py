import pandas as pd
from pandas import ExcelWriter
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

from scipy import stats

import os
from os.path import join, expanduser

import sys
sys.path.insert(0, '../libs/')
import torch
import torch.nn as nn

from ns3_dataset import NS3Dataset, Scenario
from routenet_dataset import RoutenetDataset
from understanding_dataset import UnderstandingDataset

from understanding import Understanding

import itertools
import time
from shutil import copyfile

assert len(sys.argv) == 9, "Errore"

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
- r2_test_scores.txt: list of all r2 scores
- avg_results.xlsx: one result for each search
- NN.model: model, trained not on all the training dataset,
    that provided best avg_val_loss. With EarlyStopping, it will take the model
    with the best validation, so even though it started to overfit
    I will consider the one before the beginning of the overfitting.

- for each i in range(search) -> a directory search_<i>
    - search_<i>/best_model.model: best model related to that search
    - for each k in range(cv_k) -> a directory cv_<k>
        - search_<i>/cv_<k>/description.txt: search parameters
        - search_<i>/cv_<k>/NN.model: trained model for that (s,k)
        - search_<i>/cv_<k>/loss.history: validation loss history (np array)
'''

'''
Initialize variables obtained in input and globals
'''
cache_dir = str(arguments[1])
dataset_id = str(arguments[2])
model_dir = str(arguments[3])
topology= str(arguments[4])
identifier = "simulation_{}".format(str(arguments[5]))
test_less_intensities = bool(arguments[7] == "True")
scenario = int(arguments[8])
assert scenario >= 1 and scenario <= 3, "Wrong value describing the scenario"
scenario = [Scenario.LEVEL_1, Scenario.LEVEL_2, Scenario.LEVEL_3][scenario-1]

log_epoch = False
threshold_log = 100
shuffle_all_tensors_together = False
real_case_split = False
use_PCA = False
num_threads = 8
torch.set_num_threads(num_threads)

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

'''
Import dataset plus generation of two pairs (X_<type>, y_<type>),
type E {train, test}.
When considering CV, we don't discriminate at the beginning between
validate and train, as we select at each iteration the different cases.
To do that, we define cv_k = #models we train when cross validating.
Our implementation of CV: we train the same model cv_k times, and we average the results.
What differs from normal is that we don't validate on all the dataset, as it is huge.
We fix an approximate size of the validation dataset (eg: 0.2 = test is 20%), and we pick only
cv_k validation datasets (so we are not considering all the cases).
Result: we consider X_train_cv and we select, at each iteration, the generated
indices for validation in validation_indices.

Normally, CV would create, for K=2, two datasets, 1/2 for test, 1/2 for val.
Possible improvement: retrain best found model on entire train dataset
'''

if use_ns3:
    coefficient_delay = 1000
    if "ms" in cache_dir:
        coefficient_delay = 1000
        print("INFO: Computing delay in ms...")
    test_dataset = dataset_container = NS3Dataset(also_pyg = False, scenario = scenario, generate_tensors = True, test_less_intensities = test_less_intensities, only_low = only_low, cache_dir = cache_dir, topology = topology, identifier = identifier, use_PCA = use_PCA)
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

print("INFO: Using {} dataset".format(dataset_origin))
model_dir += "_{}".format(dataset_origin)
print("INFO: model dir is {}".format(model_dir))

# save model and scores to file
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

print("INFO: Data: completed")

cv_k = 3
if real_case_split:
    print("INFO: split done by splitting a priori the capacities and intensities ALSO during cross validation")
    dataset_container.init_cv_v2(cv_k)
else:
    print("WARNING: split done by splitting validation differently from test. This could create a biased validation")
    dataset_container.init_cv(cv_k)

'''
Criterion, patience, num_epochs.
'''
criterion = nn.MSELoss()
patience = 2
num_epochs = 4096

"""
Function returning test dataset. By default it returns it as a data loader
"""
def get_test(test_as_dataloader = True):
    if test_as_dataloader:
        test_loader = dataset_container.get_test_data_loader()
    else:
        X_test, y_test = dataset_container.X_test, dataset_container.y_test
    count_test_instances = 0
    for instance in test_loader:
        if count_test_instances > 0:
            print("WARNING: test loader is batched")
        X_test, y_test = instance
        count_test_instances += 1
    y_test *= coefficient_delay

    return X_test, y_test, test_loader

"""
Initialization of input size, output size
"""
try:
    X_test, y_test, test_loader = get_test()
    print("INFO: TEST ok")
    input_size = X_test.size(1)
    output_size = y_test.size(1)

    print("INFO: Sizes of test and train input and target sets are, respectively, ({} {}), ({} {})".format(X_test.shape, y_test.shape))
except:
    print("ERROR: during taking single sets")

'''
Definition of all parameters and construction of search array of mappings.
'''
learning_rates = [0.001, 0.003, 0.005]
lambda_regs = [0.0001, 0.001, 0.005]
dropout_rates = [0]
batch_sizes = [512]
nums_hidden_layers = [4, 8, 12]
act_funs = [torch.relu]
window_sizes = [16] # useless. Kept because in other models can be useful (eg: LSTM)

hidden_sizes = [int(np.power(input_size * output_size, 0.5))]

space_search = [learning_rates, lambda_regs, dropout_rates, batch_sizes, hidden_sizes, nums_hidden_layers, act_funs]

search = []
for (learning_rate, lambda_reg, dropout_rate, batch_size, hidden_size, num_hidden_layers, act_fun) in itertools.product(*space_search):
    search.append({"lr": learning_rate, "lambda_reg": lambda_reg, "dropout_rate": dropout_rate, "batch_size":batch_size, "hidden_size": hidden_size,"num_hidden_layers": num_hidden_layers, "act_fun": act_fun})
print("INFO: space of exploration is of size {}".format(len(search)))

'''
search_result is the map containing, for each (search, cv_idx) pair, the metric history.
At the end, fixed a search, we have the history of the validation in loss array:
search_result[key]["loss"]
'''
search_result = {}
for i, s in enumerate(search):
    for k in range(cv_k):
        key = "{}_{}".format(i,k)
        search_result[key] = {"epoch": 1, "loss": [], "test": { "loss": float("+inf"), "r2": float("-inf")}}
current_search = 0

if shuffle_all_tensors_together:
    print("WARNING: you are shuffling all the tensors!")
    dataset_container.shuffle_all_tensors_together()

def validate(net, loader, info = None):
    if info is not None:
        print("INFO: {}".format(info))
    total_valid_loss = 0
    total_valid_r2 = 0
    check_validation_times = 0
    with torch.no_grad():
        for instance in loader:
            if check_validation_times > 0:
                print("WARNING: validation happened more than once!")
            net.eval()
            X, y = instance
            y_hat = net(X)
            if "est" in info:
                y_hat *= 1/coefficient_delay
            else:
                y *= coefficient_delay
            total_valid_loss = criterion(y_hat, y).item()
            total_valid_mae = mean_absolute_error(y_hat, y).item()
            total_valid_r2 = r2_score(y_hat.detach().numpy(), y.detach().numpy())
            check_validation_times += 1
    return total_valid_loss, total_valid_mae, total_valid_r2


'''
I keep track of all the r2 scores to iteratively write the result.
This allows to see whether there is something very good just by having a look
at the corresponding file.
'''
r2_test_scores = []
path_r2_test_scores = join(*[dir_output, "r2_test_scores.txt"])# run test and save result to file

def train(net, train_loader):
    delta = 0
    for j, train_instance in enumerate(train_loader):
        net.train()
        optimizer.zero_grad()
        X_train, y_train = train_instance
        start = time.time()
        y_hat = net(X_train)
        end = time.time()
        delta += end - start

        if log_epoch and j % threshold_log == 0:
            print("Instance {} out of {}. Time: {}".format(j + 1, len(train_loader), delta / threshold_log))
            delta = 0
        y_train *= coefficient_delay
        loss = criterion(y_hat, y_train)
        loss.backward()
        optimizer.step()

    return net

"""
Does entire loop of train and validate. Best network is saved on disk.
Necessary step to provide EarlyStopping (otherwise should save best_net_so_far on memory)

For each epoch, train batch by batch and update weights. You do that with the update function.
When you finish training for that epoch, you validate and ES enters the game.
- if the validation error is less than the best, curr_patience = 0, best model is saved as the current
- if the validation error is greater than the best, curr_patience++, and no best model saved
- end: if epochs > num_epochs OR curr_patience >= patience -> end of the game
"""
def train_validate(net, current_key, train_loader, val_loader):
    epoch, curr_patience, best_loss_current_k = 0, 0, float("+inf")
    while curr_patience < patience and epoch < num_epochs:
        print("INFO: Executing epoch {} out of {}".format(epoch + 1, num_epochs))

        '''
        Training loop
        '''
        net = train(net, train_loader)

        '''
        Training phase is completed. Validation.
        '''
        total_valid_loss, total_valid_mae, total_valid_r2 = validate(net, val_loader, info="Validation")

        search_result[current_key]["epoch"] = epoch
        search_result[current_key]["loss"].append(total_valid_loss)

        print("INFO: Valid loss at {}: {}\n".format(epoch, total_valid_loss))
        print("INFO: Valid R2 at {}: {}\n".format(epoch, total_valid_r2))
        print("INFO: Valid MAE at {}: {}\n".format(epoch, total_valid_mae))

        # early stopping
        if total_valid_loss < best_loss_current_k:
            print("SUCCESS: validation loss has decreased in this epoch. Saving model.")
            best_loss_current_k = total_valid_loss
            curr_patience = 0
            # copy the network in temp file
            net.save_model(
                {
                    'model': net
                },
                current_nn_cv_path)
        else:
            curr_patience += 1
            print("WARNING: No improvement, remaining patience is {}".format(patience - curr_patience))

        print("INFO: Best: {}, current: {}".format(best_loss_current_k, total_valid_loss))
        epoch += 1
    return best_loss_current_k

'''
Actual training for all searches and CVs.
'''
for i, s in enumerate(search):
        print("\nINFO: Search {} out of {}".format(i+1, len(search)))
        learning_rate = s["lr"]
        lambda_reg = s["lambda_reg"]
        dropout_rate = s["dropout_rate"]
        batch_size = s["batch_size"]
        hidden_size = s["hidden_size"]
        num_hidden_layers = s["num_hidden_layers"]
        act_fun = s["act_fun"]

        best_net_score = float("+inf")

        '''
        Init dir for models of this search
        '''
        dir_model_output_search = join(*[dir_output, "search_{}".format(i)])
        if(not os.path.isdir(dir_model_output_search)):
            os.mkdir(dir_model_output_search)

        # fixed a model I train it cv times and average the Results
        validate_results, test_results = [], []
        for k in range(cv_k):
            print("\nINFO: CV {} out of {}".format(k+1, cv_k))
            current_key = "{}_{}".format(i,k)
            '''
            Init dir for this model. This directory will contain:
            - the model
            - a log containing number of epochs, loss history, r2
            '''
            dir_model_output_cv = join(*[dir_model_output_search, "cv_{}".format(k)])
            if(not os.path.isdir(dir_model_output_cv)):
                os.mkdir(dir_model_output_cv)

            current_nn_cv_path = '{}/NN.model'.format(dir_model_output_cv)

            net = Understanding(input_size, hidden_size, output_size, dropout_rate, num_hidden_layers, act_fun)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=lambda_reg)

            if real_case_split:
                train_loader, val_loader = dataset_container.get_current_data_loaders_v2(k, batch_size)
            else:
                train_loader, val_loader = dataset_container.get_current_data_loaders(k, batch_size)

            best_loss_current_k = train_validate(net, current_key, train_loader, val_loader)

            # read best model, i.e., one before starting overfitting for <patience> iterations
            net = torch.load(current_nn_cv_path)['model'] # best model according to ES and current (k, s)

            loss, mae_test, r2_test = validate(net, test_loader, info = "Test")

            search_result[current_key]["test"]["r2"] = r2_test
            search_result[current_key]["test"]["loss"] = loss
            # save loss, epochs in log file
            description_path = '{}/description.txt'.format(dir_model_output_cv)
            # save search
            search_path = '{}/search.txt'.format(dir_model_output_cv)
            f = open(search_path, "w")
            f.write(str(s))
            f.close()
            # save loss history
            np.save('{}/loss.history'.format(dir_model_output_cv), np.array(search_result[current_key]["loss"]))
            del net
            # update and save r2_test_scores
            r2_test_scores.append(r2_test)
            f = open(path_r2_test_scores, "w")
            f.write(str(r2_test_scores))
            f.close()

            print("INFO: CV {}/{} for search {}/{} is completed. Test R2 score: {}\nTest MSE: {}\nTest MAE: {}".format(k+1, cv_k, i+1, len(search), r2_test, loss, mae_test))
            validate_results.append(best_loss_current_k)
            test_results.append(loss)

        best_index_cv = np.argmin(np.array(validate_results))
        print("From validation: Best network is {}".format(best_index_cv))
        best_index_cv = np.argmin(np.array(test_results))
        print("From test: Best network is {}".format(best_index_cv))

        best_model_cv = join(*[dir_model_output_search, "cv_{}".format(best_index_cv)])

        current_nn_cv_path = '{}/NN.model'.format(best_model_cv)

        # save best model for the current CV
        model_path = '{}/best.model'.format(dir_model_output_search)
        copyfile(current_nn_cv_path, model_path)
        print("\nSUCCESS: completed search {}/{} \n".format(i+1, len(search)))

'''
At this point search result contains all the history.
I save the validation loss in validate_avg_loss for each (search, cv)
and average for each search
'''
validate_avg_loss = []
for i,s in enumerate(search):
    all_res = []
    for k in range(cv_k):
        current_key = "{}_{}".format(i,k)
        # best from validation: all_res.append(search_result[current_key]["loss"][-1])
        all_res.append(search_result[current_key]["test"]["loss"])
    validate_avg_loss.append(np.average(np.array(all_res)))

print("INFO: best results for each search are: {}".format(validate_avg_loss))

'''
Find best network, aka, one with lower validate_avg_loss. Result: net.
Then it computes the test and saves previously described files.
'''
best_index = np.argmin(np.array(validate_avg_loss))
print("INFO: index of best solution is: {}".format(best_index))
print("INFO: parameters of the best are: {}".format(search[best_index]))
# getting best model to show test results
dir_model_output_search = join(*[dir_output, "search_{}".format(best_index)])
model_path = '{}/best.model'.format(dir_model_output_search)
net = torch.load(model_path)['model']

loss, mae_test, r2_test = validate(net, test_loader, info = "Test")
del X_test, y_test, test_loader

print("INFO: Test Loss = {}, r2: {}, {}".format(loss, r2_test, mae_test))

# save best model
model_path = '{}/NN.model'.format(dir_output)
net.save_model(
    {
    'model': net
    },
    model_path)

# grid wit one value for search. It averages results for a single search param
excel_path = '{}/avg_results.xlsx'.format(dir_output)
writer = ExcelWriter(excel_path)
df = pd.DataFrame(columns=["Parameters", "Loss", "Normalized Loss"], data={"Parameters": search, "Loss": validate_avg_loss})
df = df.sort_values(by='Loss', ascending=False)
df.to_excel(writer,'Prova')
writer.save()

# save search parameters
description_path = dir_output+"description.txt"

keys = [key for key in search[0].keys()]
search_to_file = {}
for key in keys:
    search_to_file[key] = []
for s in search:
    for key in keys:
        if s[key] not in search_to_file[key]:
            search_to_file[key].append(s[key])
f = open(description_path, "w")
f.write(str(search_to_file))
f.close()

print("INFO: AVG results excel saved in {}".format(excel_path))
print("INFO: Best model saved in {}".format(model_path))
print("INFO: Log saved in {}".format(description_path))
print("INFO: All cv results and models are saved in {}".format(dir_output))
