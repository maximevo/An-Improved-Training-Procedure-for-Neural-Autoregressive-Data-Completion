import tensorflow as tf
import math
import numpy as np
import cPickle as pickle
import time
import os
import pandas as pd

# tport prepro of log path
from tensorport import get_logs_path

flags = tf.app.flags
FLAGS = flags.FLAGS

dir_path = os.path.join( os.path.dirname( ( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )) , 'results_experiments')
PATH_TO_LOCAL_LOGS = os.path.expanduser(dir_path)

flags.DEFINE_string("logs_dir",
                    get_logs_path(root=PATH_TO_LOCAL_LOGS),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on TensorPort cloud.")
print('FLAGS.logs_dir',FLAGS.logs_dir)

scenario = None
model = None
size = None


n_random_constraints_val = 15

def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, -1)
        f.close()


class Solver:

        def __init__(self, config, model_train, train_data, val_data, test_data, my_scenario, my_model, my_size, my_dataset):
                global validation_constraints
                global universe
                global probas_d
                global n_random_constraints_val
                
                global scenario
                global model
                global size

                
                self.config = config
                self.model_train = model_train
                self.train_data = train_data
                self.val_data = val_data
                self.test_data = test_data
                self.train_writer = None
                self.val_writer = None
                self.dataset_name = my_dataset
                
                scenario = my_scenario
                model = my_model
                size = my_size
                
                
                

                unnormalized_probas_d = np.asarray([ (float)(math.factorial(self.config.timeslice_size) / (math.factorial(i)*math.factorial(self.config.timeslice_size - i))) for i in range(self.config.timeslice_size )] )
                probas_d = unnormalized_probas_d/np.sum(unnormalized_probas_d)

                if scenario == 'scenario_1':
                    # Create n_random_constraints_val random (but always the same, in order to allow comparisons...) constraints among all constraints of all sizes -> it is likely to be constraints of 'middle' size 
                    # Keep them in validation_constraints
                    validation_constraints = []
                    universe = range(self.config.timeslice_size)
                    for index,_ in enumerate(range(n_random_constraints_val)):
                        np.random.seed(seed=index)
                        size_temp = np.random.choice(self.config.timeslice_size, p = probas_d)
                        print('size_temp',size_temp)
                        np.random.seed(seed=index)
                        constraint = np.random.choice(a = universe,size =size_temp , replace = False)
                        validation_constraints.append(constraint)
                    n_random_constraints_val = len(validation_constraints)
                        
                elif scenario == 'scenario_2':
                    # Create n_random_constraints_val of pure generation: ie empty constraints 
                    # Keep them in validation_constraints
                    validation_constraints = []
                    universe = range(self.config.timeslice_size)   
                    size_temp = 0
                    constraint = np.random.choice(a = universe,size =size_temp , replace = False)
                    print('constraint',constraint)
                    validation_constraints.append(constraint)
                    n_random_constraints_val = len(validation_constraints)

                elif scenario == 'scenario_3':
                    validation_constraints = []
                    universe = range(self.config.timeslice_size)
                    for index,_ in enumerate(range(n_random_constraints_val)):
                        np.random.seed(seed=index)
                        size_temp = int(float(10*10)/(28*28)*self.config.timeslice_size)
                        print('size_temp',size_temp)
                        np.random.seed(seed=index)
                        constraint = np.random.choice(a = universe,size =size_temp , replace = False)
                        validation_constraints.append(constraint)
                    n_random_constraints_val = len(validation_constraints)

                else: 
                    raise ValueError('solver name does not match, wrong solver?')

        def get_variables(self, loss,global_step, training):

                learning_rate = tf.train.exponential_decay(
                            self.config.initial_learning_rate, global_step, self.config.decay_steps,
                            self.config.decay_rate, staircase=True)
                if self.config.update_rule == 'adam':
                        opt = tf.train.AdamOptimizer(learning_rate = learning_rate) #careful --> config must have same params than config_train and config_test except the training arg
                elif self.config.update_rule == 'momentum':
                        opt = tf.train.MomentumOptimizer(learning_rate = learning_rate)
                elif self.config.update_rule == 'rmsprop':
                        opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate)

                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, global_norm_to_print = tf.clip_by_global_norm(gradients,self.config.gradient_clip_norm)
                train_op = opt.apply_gradients(zip(clipped_gradients, params),global_step)

                if training:
                        return [loss,train_op],learning_rate

                else:
                        return loss

        def validation(self,sv,session,variables_val,n_batch_per_epoch,\
                                        best_val_error,best_epoch,n_incr_error,epoch_id,total_train_n_examples, total_train_n_1Dconds):
                """
                Run validation by computing the performance of the model on the constraints in validation_constraints.
                Compute the (residual) autoregressive product for 1 ordering: the pitch_ascending ordering ie 0..128
                Next step: extend it to the avg of K orderings
                """
                model_tmp = self.model_train
                
                # keep track of everything we compute in this run of validation 
                losses_val = np.full([self.config.timeslice_size*self.config.timeslice_size],np.nan)
                
                n_batch_per_epoch_val = int(math.ceil(self.val_data.shape[0]/self.config.batch_size_val)) + 1
                print('n_batch_per_epoch_val',n_batch_per_epoch_val)

                train_indices_val = np.arange(self.val_data.shape[0])
                # so that all models use the same random ordering of the val indices
                np.random.seed(0)
                np.random.shuffle(train_indices_val)

                total_loss_val_normalized_avg_constraints = 0
                
                for constraint_id,constraint in enumerate(validation_constraints):
                    n_known = constraint.size
                    n_unknown = self.config.timeslice_size - n_known
                    unknown_notes = np.setdiff1d(universe,constraint) # unknown notes are ordered in pitch ascending order


                    if len(set(constraint))==0:
                        temp_custom_ordering = np.array(unknown_notes) 
                    else: 
                        constraint = np.sort(constraint)
                        temp_custom_ordering = np.concatenate( [constraint , unknown_notes] ) 
                    
                    total_loss_val_normalized = 0
                    
                    for index in range(n_known,self.config.timeslice_size):
                        
                        # loop over the validation data, using batches of size self.config.batch_size_val
                        total_actual_batch_size_val = 0
                        total_loss_val_ = 0
                        
                        for i in range(n_batch_per_epoch_val):
                            start_idx_val = (i*self.config.batch_size_val)%self.val_data.shape[0]
                            idx_val = train_indices_val[start_idx_val:start_idx_val+self.config.batch_size_val]
                              
                            actual_batch_size_val = self.val_data[idx_val].shape[0]
                            
                            if actual_batch_size_val==0:
                                continue
                            total_actual_batch_size_val += actual_batch_size_val
                    
                            custom_ordering = np.tile(temp_custom_ordering,(actual_batch_size_val,1))

                            #print('index',index)
                            #print('custom_ordering',custom_ordering)
                        
                            feed_dict = {model_tmp.inputs_placeholder: self.val_data[idx_val],
                                        model_tmp.tf_d: index ,
                                        model_tmp.custom_ordering: custom_ordering}

                            loss_val_ = session.run(variables_val,feed_dict=feed_dict)

                            total_loss_val_ += np.sum(loss_val_)

                        total_loss_val_ =  float(total_loss_val_)/total_actual_batch_size_val
                        total_loss_val_normalized += total_loss_val_
                        losses_val[constraint_id*self.config.timeslice_size+index] = total_loss_val_
                    
                    # this is the avg loss per 1D conditional (we normalized by the nb of 1D conditionals involved in the conditional completion) -> this is the metric that matters for train, validation, test 
                    total_loss_val_normalized = float(total_loss_val_normalized)/(self.config.timeslice_size-n_known)
                    total_loss_val_normalized_avg_constraints += total_loss_val_normalized
                    
                # compute the mean val score before adding any other info to the dataframe
                mean_log_prob = np.nanmean(losses_val) # this depends on the sizes of the constraints chosen...
                # this is the mean of each 1D cond: multiply by timeslice size to get a number comparable to train
                mean_log_prob *= self.config.timeslice_size
                print("for info: mean_log_prob val",mean_log_prob)

                total_loss_val_normalized_avg_constraints = float(total_loss_val_normalized_avg_constraints)/n_random_constraints_val
                # this is the mean of each 1D cond: multiply by timeslice size to get a number easier to read
                total_loss_val_normalized_avg_constraints *= self.config.timeslice_size
                print("what we care about: total_loss_val_normalized_avg_constraints val",total_loss_val_normalized_avg_constraints)
                

                # losses_val is a np.array of size timeslicesize*timeslicesize: we write it in an excel file 
                output_pd = pd.DataFrame(data = np.reshape(losses_val,[1,-1]))

                # add a few metrics to keep track of      
                output_pd['time'] = time.time() - start
                output_pd['epoch_id'] = epoch_id
                output_pd['nb_train_examples div. by train_data.shape[0]'] = float(total_train_n_examples)/self.train_data.shape[0]
                output_pd['nb_1D_cond_computations div. by train_data.shape[0]*config.timeslice_size'] = float(total_train_n_1Dconds)/ (self.train_data.shape[0]*self.config.timeslice_size)

                # --- Writing to csv ---
                pathfile = os.path.join(self.config.path_plots_results, 'plots.csv')
                with open(pathfile, 'a') as f:
                        output_pd.to_csv(f, header=False,index=False)
                        f.close() 
                print('== Validation results saved in csv! ==')

                
                summary = tf.Summary()
                summary.value.add(tag='avg_val_loss_with_nb_epoch', simple_value=total_loss_val_normalized_avg_constraints)
                self.val_writer.add_summary(summary, global_step=epoch_id)

                summary_ter = tf.Summary()
                summary_ter.value.add(tag='avg_val_loss_with_nb_training_examples (divided by self.train_data.shape[0])', simple_value=total_loss_val_normalized_avg_constraints)
                self.train_writer.add_summary(summary_ter, global_step=total_train_n_examples/self.train_data.shape[0])

                summary_bis = tf.Summary()
                summary_bis.value.add(tag='avg_val_loss_with_nb_1D_cond_computations (divided by self.train_data.shape[0]*self.config.timeslice_size)', simple_value=total_loss_val_normalized_avg_constraints)
                self.val_writer.add_summary(summary_bis, global_step=total_train_n_1Dconds / (self.train_data.shape[0]*self.config.timeslice_size))
                
                # Check whether early stoppping
                if total_loss_val_normalized_avg_constraints < best_val_error:
                        best_val_error = total_loss_val_normalized_avg_constraints
                        best_epoch = epoch_id
                        n_incr_error = 0
                        sv.saver.save(session, sv.save_path, global_step=epoch_id)
                        #print('sv.global_step!!',sv.global_step)
                else:
                        n_incr_error += 1

                return best_val_error,best_epoch,n_incr_error,total_loss_val_normalized_avg_constraints


        def run_model(self, session,variables_train, learning_rate,variables_val, global_step,sv,n_epochs ):
                model_tmp = self.model_train

                # shuffle indicies
                train_indices = np.arange(self.train_data.shape[0])
                # so that all models use the same random ordering of the train indices
                np.random.seed(0)
                np.random.shuffle(train_indices)

                global_step_ = session.run(global_step)

                print('=== Starting training loop... ===')

                n_batch_per_epoch = int(math.ceil(self.train_data.shape[0]/self.config.batch_size)) + 1 #+1 to get the remaining portion of train data that doesn't fill a batch

                n_incr_error = 0
                best_val_error = np.inf
                best_train_error = np.inf
                best_train_epoch = 0
                best_val_epoch = 0

                total_train_n_examples = 0
                total_train_n_1Dconds = 0
                for epoch_id in range(self.config.n_epochs):
                        train_loss_epoch = 0
                        validation_loss_epoch = 0
                        n_examples = 0

                        for i in range(n_batch_per_epoch):

                                if sv.should_stop():
                                        break
                                if not n_incr_error < self.config.early_stop:
                                        break

                                start_idx = (i*self.config.batch_size)%self.train_data.shape[0]
                                idx = train_indices[start_idx:start_idx+self.config.batch_size]
                                
                                actual_batch_size = self.train_data[idx].shape[0]
                                if actual_batch_size==0:
                                        continue

                                if model=='model_1':
                                    d = 0
                                    ordering = np.tile(np.random.choice(range(self.config.timeslice_size), size=self.config.timeslice_size, replace=False, p=None).astype(np.int32),(actual_batch_size,1))

                                    
                                
                                elif model=='model_3':
                                    
                                    if scenario == 'scenario_1':
                                        d = np.random.choice(self.config.timeslice_size, p = probas_d)
                                    elif scenario == 'scenario_2':
                                        d = 0
                                    elif scenario == 'scenario_3':
                                        d = int(float(10*10)/(28*28)*self.config.timeslice_size)
                                    else: 
                                        raise ValueError('scenario name does not match, wrong scenario?')
                                    
                                    non_ordered = np.tile(np.random.choice(range(self.config.timeslice_size), size=self.config.timeslice_size, replace=False, p=None).astype(np.int32),(actual_batch_size,1))
                                    # order the first d-1 elements by pitch ascending order
                                    # order the last timeslice-d+1 elements by pitch ascending order
                                    lower = non_ordered[:,:d]
                                    upper = non_ordered[:,d:]
                                    lower_sorted = np.sort(lower, axis=1, kind='quicksort', order=None)
                                    upper_sorted = np.sort(upper, axis=1, kind='quicksort', order=None)
                                    ordering = np.concatenate((lower_sorted, upper_sorted), axis=1)
                                else: 
                                    raise ValueError('model name does not match, wrong model?')

                                print('d',d)
                                print('ordering',ordering)

                                feed_dict = {model_tmp.inputs_placeholder: self.train_data[idx],
                                             model_tmp.d_train : d,
                                             model_tmp.ordering_placeholder : ordering,
                                            } 

                                global_step_, variables_ = session.run([global_step, variables_train],feed_dict=feed_dict)

                                # c'est l'avg sur le batch, de la log likelihood, normalisee par le nb de 1D conditionals impliques (1), calculee grace au produit autoregressif partiel 
                                loss_train_ = variables_[0]
                                

                                n_examples += actual_batch_size
                                total_train_n_examples += actual_batch_size
                                # we compute: timeslice_size - d 1D conds!!
                                if size=='large':
                                    total_train_n_1Dconds += actual_batch_size* 1
                                elif size=='small':
                                    total_train_n_1Dconds += actual_batch_size*(self.config.timeslice_size - d)
                                else:
                                    raise ValueError('invalid size of dataset: small or large?')

                                
                                summary = tf.Summary()
                                summary.value.add(tag='Ev_minibatch_loss_train_with_nb_training_examples (divided by self.train_data.shape[0])', simple_value=loss_train_)
                                self.train_writer.add_summary(summary, global_step= total_train_n_examples/self.train_data.shape[0])
                                
                                summary_2 = tf.Summary()
                                summary_2.value.add(tag='Ev_minibatch_loss_train_with_nb_1D_cond_computations (divided by self.train_data.shape[0]*self.config.timeslice_size)', simple_value=loss_train_)
                                self.train_writer.add_summary(summary_2, global_step = total_train_n_1Dconds / (self.train_data.shape[0]*self.config.timeslice_size))

                                # c'est la somme sur le batch, de la log likelihood, normalisee par le nb de 1D conditionals impliques, calculee grace au produit autoregressif partiel -> tant qu'on boucle sur le training set, on somme toutes les likelihoods
                                train_loss_epoch += loss_train_*actual_batch_size

                        if n_examples==0:
                                continue
                        train_loss_epoch = float(train_loss_epoch)/n_examples
                        # this is the mean of each 1D cond: multiply by timeslice size to get a number easier to read
                        train_loss_epoch *= self.config.timeslice_size

                        
                        if train_loss_epoch < best_train_error:
                                best_train_error = train_loss_epoch
                                best_train_epoch = epoch_id
                        
                        summary_3 = tf.Summary()
                        summary_3.value.add(tag='Ev_epoch_loss_train_with_nb_epoch', simple_value=train_loss_epoch)
                        self.train_writer.add_summary(summary_3, global_step=epoch_id)
                        
                        summary_4 = tf.Summary()
                        summary_4.value.add(tag='Ev_epoch_loss_train_with_nb_training_examples (divided by self.train_data.shape[0])', simple_value=train_loss_epoch)
                        self.train_writer.add_summary(summary_4, global_step=total_train_n_examples/self.train_data.shape[0])


                        summary_5 = tf.Summary()
                        summary_5.value.add(tag='Ev_epoch_loss_train_with_nb_1D_cond_computations (divided by self.train_data.shape[0]*self.config.timeslice_size)', simple_value=train_loss_epoch)
                        self.train_writer.add_summary(summary_5, global_step = total_train_n_1Dconds / (self.train_data.shape[0]*self.config.timeslice_size))

                                                
                        """
                        Validation: each validation should save a ton of info into a csv file
                        Run validation each X epochs AND each Y seconds
                        """
                        if size=='large':
                            if self.dataset_name=='nips':
                                n_epochs_btw_validation = int(float(self.config.timeslice_size)/5)
                            else:
                                n_epochs_btw_validation = int(float(self.config.timeslice_size)/2)
                        elif size=='small':
                            n_epochs_btw_validation = self.config.timeslice_size/10
                        else:
                            raise ValueError('invalid size of dataset: small or large ')

                        if epoch_id % n_epochs_btw_validation == 0:

                            best_val_error,best_val_epoch,n_incr_error,total_loss_val_normalized_avg_constraints = self.validation(sv,session,variables_val,n_batch_per_epoch,\
                                    best_val_error,best_val_epoch,n_incr_error,epoch_id,total_train_n_examples, total_train_n_1Dconds)



                            print("== Epoch %d - train_loss: %.3f - val_loss: %.3f ==" %(epoch_id,train_loss_epoch,total_loss_val_normalized_avg_constraints))
                        else:
                            print("== Epoch %d - train_loss: %.3f " %(epoch_id,train_loss_epoch))


                # Save results in CSV file
                # config parameters
                parameter_keys_raw = dir(self.config)
                print('parameter_keys_raw',parameter_keys_raw)
                params_remove = ['__doc__', '__module__','__init__']
                parameter_keys = []
                for param in parameter_keys_raw:
                        if not(param in params_remove):
                                if not(param == 'dim_feature'):
                                        parameter_keys+=[param]

                # performance metrics
                performance_keys = ['best_train_error','best_train_epoch','best_val_error','best_val_epoch']

                # output csv
                output_pd = pd.DataFrame(columns = parameter_keys + performance_keys)
                # parameter columns
                for par_name in parameter_keys:
                        output_pd.set_value(0, par_name, getattr(self.config,par_name), takeable=False)
                         
                            
                # performance columns      
                output_pd['best_train_error'] = best_train_error
                output_pd['best_train_epoch'] = best_train_epoch
                output_pd['best_val_error'] = best_val_error
                output_pd['best_val_epoch'] = best_val_epoch

                # --- Writing to csv ---
                pathfile = os.path.join( self.config.path_csv_results, 'xp.csv')
                header_boolean = not(os.path.isfile(pathfile))
                with open(pathfile, 'a') as f:
                        output_pd.to_csv(f, header=header_boolean,index=False)
                print('== Results saved in csv! ==')
                print('=== Training complete. ===')

                return 


        def train(self,dataset):
                global start


                if not(self.config.id_exp == None):
                # If the config file contains the ID of the experiment, then restore this model
                        id_experiment = str(self.config.id_exp)
                else:
                # ID of the experiment is the timestamp
                        id_experiment = str(time.time()).replace('.','')
                        print('id_experiment',id_experiment)

                # Path where model will be saved


                
                log_dir = os.path.join(*[FLAGS.logs_dir,  size , dataset, model, scenario, id_experiment, 'model_saved'])
                summary_dir = os.path.join(*[FLAGS.logs_dir,  size , dataset, model, scenario, id_experiment, 'summary'])
                path_plots_results = os.path.join(*[FLAGS.logs_dir,  size , dataset, model, scenario, id_experiment, 'plots'])
                path_csv_results = os.path.join(FLAGS.logs_dir,'xp')


                print('log_dir',log_dir)
                print('summary_dir',summary_dir)
                print('path_plots_results',path_plots_results)
                print('path_csv_results',path_csv_results)

                if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                if not os.path.exists(summary_dir):
                        os.makedirs(summary_dir)
                if not os.path.exists(path_plots_results):
                        os.makedirs(path_plots_results)
                if not os.path.exists(path_csv_results):
                        os.makedirs(path_csv_results)

                self.config.log_dir = log_dir
                self.config.summary_dir = summary_dir
                self.config.path_plots_results = path_plots_results
                self.config.path_csv_results = path_csv_results

                global_step = tf.Variable(0, trainable=False)

                print('=== Creating tensorflow training graph ===')
                #if hasattr(self.model_train, 'logprob_train'):
                loss_train = self.model_train.logprob_train
                loss_val = self.model_train.logprob_val
                variables_train,learning_rate = self.get_variables(loss_train, global_step=global_step,training=True)
                variables_val = self.get_variables(loss_val, global_step=global_step,training=False)


                #else:
                #    loss = self.model_train.logprob
                #    variables_train,learning_rate = self.get_variables(loss, global_step=global_step,training=True)
                #    variables_val = self.get_variables(loss, global_step=global_step,training=False)

                # Summary op
                self.train_writer = tf.summary.FileWriter( os.path.join(self.config.summary_dir,'train') )
                self.val_writer = tf.summary.FileWriter( os.path.join( self.config.summary_dir, 'validation') )

                sv = tf.train.Supervisor(logdir=self.config.log_dir, save_model_secs=self.config.save_model_secs,
                                           global_step=global_step)

                sv.saver.max_to_keep=3
                
                print('=== Creating tensorflow session ===')
                tf.logging.set_verbosity(tf.logging.INFO)
                
                start = time.time()

                with sv.managed_session() as sess:


                        #print('== Initializing session ==')
                        #sess.run(tf.global_variables_initializer())

                        print('=== Training model ===')
                        self.run_model(sess,variables_train, learning_rate,variables_val,global_step=global_step,sv=sv,n_epochs=self.config.n_epochs)

