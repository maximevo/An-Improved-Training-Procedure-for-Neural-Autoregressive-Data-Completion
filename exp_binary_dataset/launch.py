"""
In this script, we launch the experiment.
Keywords can control: small dataset or large dataset, scenario 1,2,3 or 4, model1 or model3 
"""

import sys, os

my_path = os.path.dirname( os.path.abspath(__file__) )
root = os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))

print('my_path',my_path)
print('root',root)


if not (my_path in sys.path):
    sys.path.append(my_path)
if not (root in sys.path):
    sys.path.append(root)
    
from model import Model

from solver import Solver

from config import Config

import copy
import sys
import os
import time
import numpy as np
import mlpython.mlproblems.generic as mlpb

# ======== CHOICE OF DATASET =================
dataset_adult = 'adult'
dataset_dna = 'dna'
dataset_mushrooms = 'mushrooms'
dataset_nips = 'nips'
dataset_connect4 = 'connect4'
# ======== EXPERIMENTS TO LAUNCH =============

batch_size = 128
initial_learning_rate_1 = 0.1
initial_learning_rate_2 = 0.05
initial_learning_rate_3 = 0.01
initial_learning_rate_4 = 0.005
initial_learning_rate_5 = 0.001
initial_learning_rate_6 = 0.0005
initial_learning_rate_7 = 0.0001
initial_learning_rate_8 = 0.00005
initial_learning_rate_9 = 0.00001
initial_learning_rate_10 = 0.000005
initial_learning_rate_11 = 0.000001

decay_steps = 2000
decay_rate = 0.9
gradient_clip_norm = 9
early_stop = 20





n_epochs = 2000






config_1 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_1,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_2 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_2,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_3 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_3,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_4 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_4,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_5 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_5,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_6 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_6,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_7 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_7,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_8 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_8,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_9 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_9,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_10 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_10,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)
config_11 = Config(batch_size=batch_size,n_epochs=n_epochs,initial_learning_rate=initial_learning_rate_11,\
                       decay_steps=decay_steps,decay_rate=decay_rate,gradient_clip_norm=gradient_clip_norm,early_stop=early_stop)


#config_list = [config_1,config_2,config_3,config_4,config_5,config_6,config_7,config_8,config_9,config_10,config_11]
config_list = [config_4,config_5,config_6]

#size_datasets = ['small','large']
size_datasets = ['small','large']

#models = ['model_1','model_3']
models = ['model_1','model_3']

#dataset_list = [dataset_adult, dataset_dna, dataset_mushrooms, dataset_nips]
dataset_list = [dataset_adult]

#scenarios = ['scenario_1','scenario_2','scenario_3']
scenarios = ['scenario_1','scenario_2','scenario_3']

xp_to_launch = []

for dataset in dataset_list:
    for my_size in size_datasets:
        for config in config_list:
            for my_scenario in scenarios:
                for my_model in models:
                    xp_to_launch.append( (Model,dataset,config, Solver, my_scenario, my_model, my_size ) )

                
print('xp_to_launch',xp_to_launch)                


# ======== END EXPERIMENTS TO LAUNCH =============


def load_data(dataset_name):
    datadir = root + '/data/'
    exec 'import mlpython.datasets.'+dataset_name+' as mldataset'
    exec 'datadir = datadir + \''+dataset_name+'/\''
    all_data = mldataset.load(datadir,load_to_memory=True)
    train_data, train_metadata = all_data['train']
    if dataset_name == 'binarized_mnist' or dataset_name == 'nips': 
        trainset = mlpb.MLProblem(train_data,train_metadata)
    else:
        trainset = mlpb.SubsetFieldsProblem(train_data,train_metadata)
    trainset.setup()
    valid_data, valid_metadata = all_data['valid']
    validset = trainset.apply_on(valid_data,valid_metadata)
    test_data, test_metadata = all_data['test']
    testset = trainset.apply_on(test_data,test_metadata)

    train_X = trainset.data.mem_data[0]
    valid_X = validset.data.mem_data[0]
    test_X = testset.data.mem_data[0]
    return train_X,valid_X,test_X


def launch_one_experiment(model,dataset,config, my_solver, my_scenario, my_model, my_size):

    datasets = ['adult', 'binarized_mnist', 'connect4', 'dna', 'mushrooms', 'nips', 'ocr_letters', 'rcv1', 'web']
    if dataset not in datasets:
        raise ValueError('dataset '+dataset+' unknown')

    # Load dataset
    print('Loading dataset '  + dataset)
    train_X,valid_X,test_X = load_data(dataset)
    print("Data Loaded")

    print('train_X.shape',train_X.shape)
    print('valid_X.shape',valid_X.shape)
    print('test_X.shape',test_X.shape)


    # Instantiate config object
    config.timeslice_size = train_X.shape[1]

    # Define models
    model_train = model(config, my_size = my_size)
    print "Train model instantiated"

    # Instantiate solver
    solver = my_solver(config, model_train, train_X, valid_X, test_X, my_scenario = my_scenario, my_model = my_model, my_size = my_size, my_dataset = dataset)
    print "Solver instantiated, starting training"
    
    # Train model
    solver.train(dataset)

def main(): 
    for xp_id, (model,dataset,config, solver, my_scenario, my_model, my_size) in enumerate(xp_to_launch):
        print('===== Launching experiment '+ str(xp_id+1) + " =====")

        model_name = str(model).split('.')[-1]
        print('model: ', model_name, my_model, 'dataset: ',dataset,'config: ',config)
        launch_one_experiment(model,dataset,config, solver, my_scenario, my_model, my_size)

if __name__ == "__main__":
    main()
