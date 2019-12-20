from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

import os
from os.path import isfile, join,expanduser

import sys
sys.path.insert(0, '../libs/')
import torch
from torch import nn

from ns3_dataset import NS3Dataset, Scenario
from routenet_dataset import RoutenetDataset
from understanding_dataset import UnderstandingDataset

from torch_geometric.data import DataLoader
import itertools
import time
from shutil import copyfile

from gnn_simple import Net as GCNet
from gnn_L2 import Net as GConvNet

import itertools
import time
import pathlib
import numpy as np

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
num_threads = 16
torch.set_num_threads(num_threads)

batch_sizes = [512]
batch_size = batch_sizes[0]
FACTOR_DELAY = 1000

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
    if "ms" in cache_dir:
        print("INFO: Computing delay in ms...")
    dataset_container = NS3Dataset(also_pyg = True, scenario = scenario, generate_tensors = True, test_less_intensities = test_less_intensities, only_low = only_low, cache_dir = cache_dir, topology = topology, identifier = identifier, use_PCA = use_PCA)

    number_of_train_val_simulations = len(dataset_container) / batch_size
    print("number_of_train_val_simulations {}".format(number_of_train_val_simulations))
    number_of_train_val_simulations = int(number_of_train_val_simulations)
    number_of_train_simulations = int(number_of_train_val_simulations * 8 / 10)
    train_dataset = dataset_container[:int(number_of_train_simulations * batch_size)]
    val_dataset = dataset_container[len(train_dataset):]
    del dataset_container
    print("INFO: train and validate are loaded and shuffled")

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

'''
Criterion, patience, num_epochs.
'''
criterion = nn.MSELoss()
patience = 2
num_epochs = 4096

'''
Definition of all parameters and construction of search array of mappings.
'''
learning_rate = 0.003
lambda_reg = 0.0003

train_dataset_length = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

del train_dataset, val_dataset

def train():
    model.train()

    loss_all = 0
    num_iter = 0
    log_epoch = True
    log_step = 512
    start = time.time()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        try:
            hidden_state = model.init_hidden(data.x.size(0))
            output = model(data, hidden_state)
        except:
            output = model(data)
        y_train = data.y.to(device) * FACTOR_DELAY
        loss = crit(output, y_train.view(-1, output.shape[1]))
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        num_iter += 1
        if log_epoch and num_iter % log_step == 0:
            end = time.time()
            print("Instance {} out of {}. Time: {}".format(num_iter+1, len(train_loader), end - start))
            start = time.time()
    return loss_all / train_dataset_length

def evaluate(loader, info = None):
    model.eval()

    predictions = []
    y_val = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            try:
                hidden_state = model.init_hidden(data.x.size(0))
                pred = model(data, hidden_state)
            except:
                pred = model(data)

            if info is not None and "est" in info:
                y_real = data.y.view(-1, pred.shape[1])
                pred *= 1 / FACTOR_DELAY
            else:
                y_real = data.y.view(-1, pred.shape[1]) * FACTOR_DELAY

            predictions.append(pred)
            y_val.append(y_real)
    y_val = torch.cat(y_val).detach().numpy()
    predictions = torch.cat(predictions).detach().numpy()

    return r2_score (y_val, predictions), mean_squared_error(y_val, predictions)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
if scenario == Scenario.LEVEL_1:
    model = GCNet().to(device)
else:
    model = GConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)
crit = torch.nn.MSELoss()
prev_val_r2 = float("-inf")
prev_val_mse = float("+inf")
val_r2 = 0
val_mse = float("+inf")
epoch = 1
while prev_val_mse >= val_mse and epoch < num_epochs:
    c_loss = train()
    prev_val_r2 = val_r2
    prev_val_mse = val_mse
    val_r2, val_mse = evaluate(val_loader, info="Validate")
    print("Epoch {}.\n\nR2\nValidate: {}\n\nMSE\nValidate: {}\n".format(epoch, val_r2, c_loss, val_mse))
    epoch += 1

del train_loader, val_loader

print("SUCCESS: training is done.\nINFO: Start testing...")
test_dataset = NS3Dataset(only_test=True, also_pyg=True, scenario=scenario, generate_tensors=True,
                          test_less_intensities=test_less_intensities, only_low=only_low, cache_dir=cache_dir,
                          topology=topology, identifier=identifier, use_PCA=use_PCA)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
print("INFO: test is loaded")

# save model

model_path = '{}/pyg.model'.format(dir_output)
model.save_model(
    {
    'model': model
    },
    model_path)

test_r2, test_mse = evaluate(test_loader, info="Test")

del test_loader

print("Test: MSE = {}, R2 = {}".format(test_mse, test_r2))
log_path = join(*[dir_output, "log.txt"])

output_log = "Model: {}\nMSE: {}\nR2: {}".format(model_path, test_mse, test_r2)
f = open(log_path, "w")
f.write(output_log)
f.close()

print("Test result provided at \n{}".format(log_path))
