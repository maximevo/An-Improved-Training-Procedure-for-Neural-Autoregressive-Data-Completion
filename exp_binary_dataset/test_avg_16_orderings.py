"""
In this script, we launch the experiment.
Keywords can control: small dataset or large dataset, scenario 1,2,3 or 4, model1 or model3 
"""


###########
# BEGIN TO CHOOSE IN ORDER TO LAUNCH TEST!
###########
# Define size hidden layer
size_hidden_layer = 500
batch_size_test = 512

xp_to_launch = []


xp_to_launch = [
   
('/home/maximev/results_experiments_avg_16_ordering/large/adult/model_1/scenario_1/150411199818/model_saved','model_1', 'adult', 'scenario_1'),
    ('/home/maximev/results_experiments_avg_16_ordering/large/adult/model_3/scenario_1/150411259396/model_saved','model_3', 'adult', 'scenario_1'),
    
    ('/home/maximev/results_experiments_avg_16_ordering/large/adult/model_1/scenario_2/150411321747/model_saved','model_1', 'adult', 'scenario_2'),
    ('/home/maximev/results_experiments_avg_16_ordering/large/adult/model_3/scenario_2/150411382132/model_saved','model_3', 'adult', 'scenario_2'),
    
    ('/home/maximev/results_experiments_avg_16_ordering/large/adult/model_1/scenario_3/150411446331/model_saved','model_1', 'adult', 'scenario_3'),
    ('/home/maximev/results_experiments_avg_16_ordering/large/adult/model_3/scenario_3/150411526819/model_saved','model_3', 'adult', 'scenario_3'),
    
    
    
    
    
    
    ('/home/maximev/results_experiments_avg_16_ordering/large/dna/model_1/scenario_1/150411669833/model_saved','model_1', 'dna', 'scenario_1'),
    ('/home/maximev/results_experiments_avg_16_ordering/large/dna/model_3/scenario_1/150411315163/model_saved','model_3', 'dna', 'scenario_1'),
    
    ('/home/maximev/results_experiments_avg_16_ordering/large/dna/model_1/scenario_2/15041179954/model_saved','model_1', 'dna', 'scenario_2'),
    ('/home/maximev/results_experiments_avg_16_ordering/large/dna/model_3/scenario_2/150411428912/model_saved','model_3', 'dna', 'scenario_2'),
    
    ('/home/maximev/results_experiments_avg_16_ordering/large/dna/model_1/scenario_3/150411897736/model_saved','model_1', 'dna', 'scenario_3'),
    ('/home/maximev/results_experiments_avg_16_ordering/large/dna/model_3/scenario_3/150411993065/model_saved','model_3', 'dna', 'scenario_3'),
    
    
    
    
]


###########
# END TO CHOOSE IN ORDER TO LAUNCH TEST!
###########





import sys, os
import math
import tensorflow as tf
import pandas as pd

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






dirfile = './../../results_experiments/test/'

if not os.path.exists(dirfile):
    os.makedirs(dirfile)

pathfile = dirfile + 'test_xp.csv'





for (path_to_restore,model,dataset,scenario) in xp_to_launch:



    # load dataset
    datasets = ['adult', 'binarized_mnist', 'connect4', 'dna', 'mushrooms', 'nips', 'ocr_letters', 'rcv1', 'web']
    if dataset not in datasets:
        raise ValueError('dataset '+dataset+' unknown')

    # Load dataset
    print('Loading dataset '  + dataset)
    _, _, test_X = load_data(dataset)
    print("Data Loaded")

    print('test_X.shape',test_X.shape)

    # Choose timeslice_size
    timeslice_size = test_X.shape[1]



    # Create the 16 orderings that we will average
    mp_16 = []
    sorting_dict_16 = []

    universe = range(timeslice_size)   
    for index in range(16):
        np.random.seed(seed=index*index*index+19)
        ordering_list = list(np.random.choice(a = universe,size = timeslice_size , replace = False))
        sorting_dict = dict(zip(ordering_list, universe))

        mp = np.arange(0,timeslice_size)
        mp[sorting_dict.keys()] = sorting_dict.values()

        mp_16.append(mp)
        sorting_dict_16.append(sorting_dict)

        
    # Create test constraints: choose a different random seed than validation
    unnormalized_probas_d = np.asarray([ (float)(math.factorial(timeslice_size) / (math.factorial(i)*math.factorial(timeslice_size - i))) for i in range(timeslice_size )] )
    probas_d = unnormalized_probas_d/np.sum(unnormalized_probas_d)

    n_random_constraints_test = 20

    if scenario == 'scenario_1':
                        # Create n_random_constraints_test random (but always the same, in order to allow comparisons...) constraints among all constraints of all sizes -> it is likely to be constraints of 'middle' size 
                        # Keep them in validation_constraints
                        test_constraints = []
                        universe = range(timeslice_size)
                        for index,_ in enumerate(range(n_random_constraints_test)):
                            np.random.seed(seed=index*index+1)
                            size_temp = np.random.choice(timeslice_size, p = probas_d)
                            print('size_temp',size_temp)
                            np.random.seed(seed=index*index+1)
                            constraint = np.random.choice(a = universe,size =size_temp , replace = False)
                            test_constraints.append(constraint)

    elif scenario == 'scenario_2':
                        # Create n_random_constraints_test of uniformly random size (but always the same, in order to allow comparisons...) 
                        # Keep them in validation_constraints
                        test_constraints = []
                        universe = range(timeslice_size)
                        for index,_ in enumerate(range(n_random_constraints_test)):
                            np.random.seed(seed=index*index+1)
                            size_temp = np.random.choice(timeslice_size)
                            print('size_temp',size_temp)
                            np.random.seed(seed=index*index+1)
                            constraint = np.random.choice(a = universe,size =size_temp , replace = False)
                            test_constraints.append(constraint)

    elif scenario == 'scenario_3':
                        test_constraints = []
                        universe = range(timeslice_size)
                        for index,_ in enumerate(range(n_random_constraints_test)):
                            np.random.seed(seed=index*index+1)
                            size_temp = int(float(10*10)/(28*28)*timeslice_size)
                            print('size_temp',size_temp)
                            np.random.seed(seed=index*index+1)
                            constraint = np.random.choice(a = universe,size =size_temp , replace = False)
                            test_constraints.append(constraint)

    else: 
                        raise ValueError('solver name does not match, wrong solver?')





    def my_func_3(ordering):
                #print('shape my func 3',np.asarray(range(ordering.shape[0]),dtype=np.int32).shape)
                return np.asarray(range(ordering.shape[0]),dtype=np.int32)

    def my_func_4(ordering):
                # we can probably do it in tensorflow graph! -> see an assignment of CS231N, they have the line of code doing this ! (assignmnet 2 or 3)
                return np.asarray([0]*ordering.shape[0],dtype=np.int32)

    def get_mask_float(ordering,x):
                # NOT EFFICIENT IMPLEMENTATION!!!!!
                mask = np.zeros_like(ordering,dtype = np.float32)
                col_idx = ordering[:,:x]
                dim_1_idx = np.array(range(ordering.shape[0]))
                mask[dim_1_idx[:, None], col_idx]=1
                return mask

    tf.reset_default_graph()

    #placeholders : very close to validation graph!
    inputs_placeholder = tf.placeholder(tf.float32,shape=(None,timeslice_size))
    # tf_d is used for validation
    tf_d = tf.placeholder(tf.int32, shape=[])
    custom_ordering_tf = tf.placeholder(tf.int32,shape = [None,timeslice_size])


    my_ordering = custom_ordering_tf

    # Compute validation: for a constraint of size k, we compute the autoregressive product p(x_unknwn | x_knwn) in pitch-ascending ordering of x_unknwn. Note: the ordering is provided to this script: it is a placeholder

    with tf.variable_scope('OrderlessNADE_model') as scope:
        W = tf.get_variable("W", shape = (2*timeslice_size, size_hidden_layer), initializer = tf.contrib.layers.xavier_initializer())
        V = tf.get_variable("V", shape = (size_hidden_layer, timeslice_size), initializer = tf.contrib.layers.xavier_initializer())

        b = tf.get_variable("b",shape=(1,timeslice_size) ,dtype=np.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
        a = tf.get_variable("a",shape=(1,size_hidden_layer) ,dtype=np.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)

        scope.reuse_variables()

    inputs_flat = inputs_placeholder


    # Offset so that the loss is never equal to log(0)
    offset = tf.constant(10**(-14), dtype=tf.float32,name='offset', verify_shape=False)

    log_probability_test = 0

    with tf.variable_scope("OrderlessNADE_step"):
                #temp_mask = tf.ones_like(inputs_flat, dtype=tf.float32)
                row_indices_test = tf.py_func(my_func_3, [my_ordering], tf.int32)
                row_indices_b_test = tf.py_func(my_func_4, [my_ordering], tf.int32)

                o_d_test = my_ordering[:,tf_d]
                #print('o_d_val',o_d_val)

                temp_mask_test = tf.py_func(get_mask_float, [my_ordering,tf_d], tf.float32)

                inputs_flat_masked_test = inputs_flat*temp_mask_test

                # Replace by RELU??
                hi_test = tf.sigmoid(tf.matmul(tf.concat([inputs_flat_masked_test, temp_mask_test], 1) ,W)+a)

                coords_test = tf.transpose(tf.stack([row_indices_test, o_d_test]))

                #temp_b = b[0,d]
                coords_b_test = tf.transpose(tf.stack([row_indices_b_test, o_d_test]))
                temp_b_test =  tf.gather_nd(b, coords_b_test)

                # method 1: if each element of the batch has a different o_d
                #p_shape_val = tf.shape(V)
                #p_flat_val = tf.reshape(V, [-1])
                #i_temp_val = tf.reshape(tf.range(0, p_shape_val[0]) * p_shape_val[1], [1, -1])

                #o_d_temp_val = tf.reshape(o_d_val,[-1,1])
                #i_flat_val = tf.reshape( i_temp_val + tf.reshape(o_d_val,[-1,1]), [-1])

                #before_Z_val = tf.gather(p_flat_val, i_flat_val) # WHY this intermediate result is 1 line rather than multiple lines? 
                #Z_val =  tf.reshape(before_Z_val, [-1,p_shape_val[0]] ) # Z_0 SHOULD HAVE THE SAME LINES !!
                # DOES Z CONTAIN THE VALUES WE EXPECT?? especially the last reshape: not sure it gives the expected result ...
                #alternative_temp_product_val = hi_val*Z_val
                #temp_product_val = tf.reduce_sum( alternative_temp_product_val, 1)


                # method 2: if all the elements of the batch have the same o_d
                V_o_d_test = V[:,o_d_test[0]]
                V_o_d_test = tf.reshape(V_o_d_test,[-1,1])
                temp_product_alternative_test = tf.matmul(hi_test,V_o_d_test)
                temp_product_alternative_test = tf.reshape(temp_product_alternative_test,[-1,])

                p_o_d_test=tf.sigmoid(temp_b_test + temp_product_alternative_test)
                v_o_d_test = tf.gather_nd(inputs_flat, coords_test)

                log_prob_test = tf.multiply(v_o_d_test,tf.log(p_o_d_test + offset)) + tf.multiply((1-v_o_d_test),tf.log((1-p_o_d_test) + offset))   ## shape = (?, 1) 1 probability for each element in the batch an for all time

                #log_probability_val += log_prob_val

                #loss_val = -tf.reduce_mean(log_probability_val)
                loss_test = -log_prob_test





    saver = tf.train.Saver()

    # Create session
    session = tf.Session()
    # Restore parameters

    # Choose exp ID 
    ckpt_path = tf.train.latest_checkpoint(path_to_restore)
    print('=================== ckpt_path ===============',ckpt_path)
    saver.restore(session, ckpt_path)


    losses_test = np.full([timeslice_size*timeslice_size],np.nan)


    n_batch_per_epoch_test = int(math.ceil(test_X.shape[0]/batch_size_test)) + 1
    print('n_batch_per_epoch_test',n_batch_per_epoch_test)

    train_indices_test = np.arange(test_X.shape[0])

    np.random.seed(0)
    np.random.shuffle(train_indices_test)

    total_loss_test_normalized_avg_constraints = 0


    for constraint_id,constraint in enumerate(test_constraints):



                        n_known = constraint.size
                        n_unknown = timeslice_size - n_known
                        unknown_notes = np.setdiff1d(universe,constraint) # unknown notes are ordered in pitch ascending order
                        print('constraint_id',constraint_id)

                        for j in range(16):
                            sorting_dict = sorting_dict_16[j]

                            unknown_notes_sorted = sorted(unknown_notes,key = lambda x: sorting_dict[x])

                            constraint_sorted = sorted(constraint,key = lambda x: sorting_dict[x])

                            if len(set(constraint))==0:
                                temp_custom_ordering = np.array(unknown_notes) 
                            else: 
                                constraint = np.sort(constraint)
                                temp_custom_ordering = np.concatenate( [constraint , unknown_notes] ) 

                            total_loss_test_normalized = 0

                            for index in range(n_known,timeslice_size):

                                total_actual_batch_size_test = 0
                                total_loss_test_ = 0

                                for i in range(n_batch_per_epoch_test):

                                    start_idx_test = (i*batch_size_test)%test_X.shape[0]
                                    idx_test = train_indices_test[start_idx_test:start_idx_test+batch_size_test]


                                    actual_batch_size_test = test_X[idx_test].shape[0]


                                    if actual_batch_size_test==0:
                                        continue
                                    total_actual_batch_size_test += actual_batch_size_test

                                    custom_ordering = np.tile(temp_custom_ordering,(actual_batch_size_test,1))

                                    feed_dict = {inputs_placeholder: test_X[idx_test],
                                                tf_d: index ,
                                                custom_ordering_tf: custom_ordering}

                                    loss_test_ = session.run(loss_test,feed_dict=feed_dict)


                                    total_loss_test_ += np.sum(loss_test_)

                                total_loss_test_ =  float(total_loss_test_)/total_actual_batch_size_test
                                total_loss_test_normalized += total_loss_test_

                                losses_test[constraint_id*timeslice_size+index] = total_loss_test_
  
                            # this is the avg loss per 1D conditional (we normalized by the nb of 1D conditionals involved in the conditional completion) -> this is the metric that matters for train, validation, test 
                            total_loss_test_normalized = float(total_loss_test_normalized)/(timeslice_size-n_known)
                            total_loss_test_normalized_avg_constraints += total_loss_test_normalized

    total_loss_test_normalized_avg_constraints = float(total_loss_test_normalized_avg_constraints)/ (16.*n_random_constraints_test)
    # this is the mean of each 1D cond: multiply by timeslice size to get a number easier to read
    total_loss_test_normalized_avg_constraints *= timeslice_size
    print("what we care about: total_loss_test_normalized_avg_constraints val over 16 constraints ", total_loss_test_normalized_avg_constraints)

    losses_val = losses_val/16.
    
    
    output_pd = pd.DataFrame(np.reshape([path_to_restore,total_loss_test_normalized_avg_constraints],[1,2]),columns=['path','test score'])
    

    print('output_pd',output_pd)
    
    header_boolean = not(os.path.isfile(pathfile))
    with open(pathfile, 'a') as f:
        output_pd.to_csv(f, header=header_boolean,index=False)


                        
