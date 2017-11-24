"""
With this script, we can do small (estimator of the loss) and large (full loss) models, depending on the value of global keywords
"""
import numpy as np
import tensorflow as tf
size = None

class Model:
	def __init__(self, config, my_size):
		global size
		size = my_size
		tf.reset_default_graph()
		self.config = config
		self.inputs_placeholder = None
		self.dropout_placeholder = None
		self.build()

	def add_placeholders(self):
		self.inputs_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.timeslice_size))
       		# d_train is used for training
		self.d_train = tf.placeholder(tf.int32, shape=[])
		self.ordering_placeholder = tf.placeholder(tf.int32,shape = [None, self.config.timeslice_size])
        	# tf_d is used for validation
		self.tf_d = tf.placeholder(tf.int32, shape=[])
		self.custom_ordering = tf.placeholder(tf.int32,shape = [None,self.config.timeslice_size])

	def compute_row_indices(self,ordering):
		return np.asarray(range(ordering.shape[0]),dtype=np.int32)

	def compute_row_indices_b(self,ordering):
		return np.asarray([0]*ordering.shape[0],dtype=np.int32)

	def get_mask_float(self,ordering,x):
		mask = np.zeros_like(ordering,dtype = np.float32)
		col_idx = ordering[:,:x]
		dim_1_idx = np.array(range(ordering.shape[0]))
		mask[dim_1_idx[:, None], col_idx]=1
		return mask

	def get_o_d_different_i(self,ordering,x):
		return ordering[range(len(ordering)),x]

	def get_mask_float_different_i(self,ordering,x):
		mask = np.zeros_like(ordering,dtype = np.float32)
		i_list_max = np.max(x)
		indices_array = np.tile(np.arange(i_list_max),(ordering.shape[0],1))
		bools = (indices_array >= x)
		col_idx = ordering[:,:i_list_max]
		col_idx[bools]=ordering[0][0]
		dim_1_idx = np.array(range(ordering.shape[0]))
		mask[dim_1_idx[:, None], col_idx]=1
		return mask

	def get_log_prob_op(self, training=False):
		""""
		This function defines tf graph that computes the loss.
		There are 128 vectors in the batch. We compute p(x_unknown | x_known) under the model. 
		To do so, we compute 1 or all the terms of the auto-regressive product corresponding to x_unknown. The partial auto-regressive product is computed in the pitch-ascneding order of the unknown notes      
		Validation is done by computing 1 partial likelihood of size k for each k (ie one precise constraint of size k) (ie the partial autoregressive product) under the model, in the pitch-ascending ordering (not in a random ordering!) 
		The partial auto-regressive products are computed in the same (pitch ascending) order
		"""
		with tf.variable_scope('OrderlessNADE_model') as scope:
			W = tf.get_variable("W", shape = (2*self.config.timeslice_size, self.config.size_hidden_layer), initializer = tf.contrib.layers.xavier_initializer())
			V = tf.get_variable("V", shape = (self.config.size_hidden_layer, self.config.timeslice_size), initializer = tf.contrib.layers.xavier_initializer())
			b = tf.get_variable("b",shape=(1,self.config.timeslice_size) ,dtype=np.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
			a = tf.get_variable("a",shape=(1,self.config.size_hidden_layer) ,dtype=np.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
			scope.reuse_variables()
		inputs_flat = self.inputs_placeholder
		# Offset so that the loss is never equal to log(0)
		offset = tf.constant(10**(-14), dtype=tf.float32,name='offset', verify_shape=False)
		# For each element of the batch, we create the ordering provided by the user. It should be made of the known note (eg in pitch ascending order but it doesnt matter) followed by the unknown notes in pitch ascending order.
        	# Thanks to these placeholders, it should work for both Model1 and Model3
		d = self.d_train
        	# ordering_placeholder should be provided as shape: (x.shape[0],timeslice_size): no need to tile it
		ordering = self.ordering_placeholder
		#ordering = tf.py_func(self.my_func_custom_ordering, [inputs_flat,temp_train_ordering], tf.int32)        
		# Intialize loss to 0
		log_probability_train = tf.zeros([tf.shape(inputs_flat)[0],], dtype=tf.float32, name=None)

		with tf.variable_scope("OrderlessNADE_step"):
			# Useful later on to slice tensors in tf graph
			row_indices = tf.py_func(self.compute_row_indices, [ordering], tf.int32)
			row_indices_b = tf.py_func(self.compute_row_indices_b, [ordering], tf.int32)

            		# index to make sure we compute only one term of the autoregressive product
			index = tf.constant(0)
         		   # start_index: the index of the term of the autoregressive product that we compute
			if size=='large':
				start_index = tf.random_uniform(shape=[tf.shape(inputs_flat)[0],1], minval=d,maxval=self.config.timeslice_size, dtype=tf.int32) 
				while_condition = lambda log_probability_train,i,start_i, reshaped_start_i: tf.less(i, 1)
			elif size=='small':
				#start_index = tf.constant(d, dtype=tf.int32, name='trash_name', verify_shape=False) # shape = [tf.shape(inputs_flat)[0],1] 
				start_index = tf.tile(tf.reshape(d, [1,1]), multiples= (tf.shape(inputs_flat)[0],1) ) # shape = [tf.shape(inputs_flat)[0],1] # tf.cast(d,tf.int32)
				while_condition = lambda log_probability_train,i,start_i, reshaped_start_i: tf.less(i, self.config.timeslice_size - d)
			else:
				raise ValueError('invalid shape provided to the Model')
                
			reshaped_start_index = tf.reshape(start_index,[-1,])


			def body(log_probability_train,i, start_i, reshaped_start_i):
				# each element of the batch has it's own ordering: we pick the d+1-th element
				o_d = tf.py_func(self.get_o_d_different_i, [ordering,reshaped_start_i], tf.int32)

				# each element of the batch has a mask: equal to 1 for the first d variables of the ordering, 0 for the other variables
				#temp_mask = tf.py_func(self.get_mask_float, [ordering,start_i], tf.float32)
				temp_mask = tf.py_func(self.get_mask_float_different_i, [ordering,start_i], tf.float32) # CAN BE DONE IN SOLVER!!!!!

				# mask out the input 
				inputs_flat_masked = inputs_flat*temp_mask

				# concat masked inputs and mask
				# pass it thought feed-forward neural network
				# replace sigmoid non-linearity by RELU??
				hi = tf.sigmoid(tf.matmul(tf.concat([inputs_flat_masked, temp_mask], 1) ,W)+a)

				coords = tf.transpose(tf.stack([row_indices, o_d]))
				coords_b = tf.transpose(tf.stack([row_indices_b, o_d]))
				temp_b =  tf.gather_nd(b, coords_b)
                
				p_shape = tf.shape(V)
				p_flat = tf.reshape(V, [-1])
				i_temp = tf.reshape(tf.range(0, p_shape[0]) * p_shape[1], [1, -1])
				o_d_temp = tf.reshape(o_d,[-1,1])
				i_flat = tf.reshape( i_temp + tf.reshape(o_d,[-1,1]), [-1])
				before_Z = tf.gather(p_flat, i_flat) 
				Z =  tf.reshape(before_Z, [-1,p_shape[0]] ) 
				alternative_temp_product = hi*Z
				temp_product = tf.reduce_sum( alternative_temp_product, 1)
				p_o_d=tf.sigmoid(temp_b + temp_product)
				v_o_d = tf.gather_nd(inputs_flat, coords)
				log_prob = tf.multiply(v_o_d,tf.log(p_o_d + offset)) + tf.multiply((1-v_o_d),tf.log((1-p_o_d) + offset))
				### shape = (?, 1) 1 probability for each element in the batch an for all time
				log_prob = tf.reshape(log_prob, (tf.shape(log_prob)[0],) )
				log_probability_train += log_prob
				return [log_probability_train,tf.add(i, 1), tf.add(start_i, 1), tf.add(reshaped_start_i, 1)]
		# loop over the autoregressive sum
		log_probability_train,index_after_loop, _, _ = tf.while_loop(while_condition, body, [log_probability_train,index,start_index, reshaped_start_index]) #shape_invariants
		# need to divide by the number of 1D conditional terms comuted: 
		log_probability_train = log_probability_train/tf.cast(index_after_loop, tf.float32)
		loss_train = -tf.reduce_mean(log_probability_train)
		my_ordering = self.custom_ordering
        
		# Compute validation: for a constraint of size k, we compute the autoregressive product p(x_unknwn | x_knwn) in pitch-ascending ordering of x_unknwn. Note: the ordering is provided to this script: it is a placeholder
		log_probability_val = 0
		with tf.variable_scope("OrderlessNADE_step"):
		    #temp_mask = tf.ones_like(inputs_flat, dtype=tf.float32)
			row_indices_val = tf.py_func(self.compute_row_indices, [my_ordering], tf.int32)
			row_indices_b_val = tf.py_func(self.compute_row_indices_b, [my_ordering], tf.int32)

			tf_d = self.tf_d
            		#for d in range(self.config.timeslice_size):
			o_d_val = my_ordering[:,tf_d]
			temp_mask_val = tf.py_func(self.get_mask_float, [my_ordering,tf_d], tf.float32)
			inputs_flat_masked_val = inputs_flat*temp_mask_val
			# Replace by RELU??
			hi_val = tf.sigmoid(tf.matmul(tf.concat([inputs_flat_masked_val, temp_mask_val], 1) ,W)+a)
			coords_val = tf.transpose(tf.stack([row_indices_val, o_d_val]))
			#temp_b = b[0,d]
			coords_b_val = tf.transpose(tf.stack([row_indices_b_val, o_d_val]))
			temp_b_val =  tf.gather_nd(b, coords_b_val)

			V_o_d_val = V[:,o_d_val[0]]
			V_o_d_val = tf.reshape(V_o_d_val,[-1,1])
			temp_product_alternative_val = tf.matmul(hi_val,V_o_d_val)
			temp_product_alternative_val = tf.reshape(temp_product_alternative_val,[-1,])

			p_o_d_val=tf.sigmoid(temp_b_val + temp_product_alternative_val)
			v_o_d_val = tf.gather_nd(inputs_flat, coords_val)

			log_prob_val = tf.multiply(v_o_d_val,tf.log(p_o_d_val + offset)) + tf.multiply((1-v_o_d_val),tf.log((1-p_o_d_val) + offset))   ## shape = (?, 1) 1 probability for each element in the batch an for all time
			loss_val = -log_prob_val
		return loss_train,loss_val,d

	def build(self):
		print 'Tensorflow version: '
		print(tf.__version__)
		self.add_placeholders()
		self.logprob_train,self.logprob_val,self.d = self.get_log_prob_op()
