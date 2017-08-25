import os

'''
	Config object class.
	We feed a config object to the model and solver builder.
'''

class Config:
	'''
		Config for ED model
	'''

	def __init__(self,size_hidden_layer=500,num_threads=2,batch_size=128,batch_size_val=512,n_epochs=2000,initial_learning_rate=0.005,decay_steps=1000, \
					decay_rate = 0.99,gradient_clip_norm=5,update_rule='adam',early_stop=20,path_to_saved_model=None):

		# TODO: Complete with all default parameters

		# Data loading parameters
		self.timeslice_size = None


		# Model parameters
		self.size_hidden_layer = size_hidden_layer
		self.num_threads = num_threads
		self.batch_size = batch_size
		self.batch_size_val = batch_size_val

		# Training parameters
		self.n_epochs = n_epochs
		self.initial_learning_rate = initial_learning_rate
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.gradient_clip_norm = gradient_clip_norm
		self.update_rule = update_rule
		self.early_stop = early_stop # number of epochs before early stopping

		# Saving parameters
		self.id_exp = None
		self.log_dir = None
		self.path_csv_results = None
		self.path_plots_results = None
		self.save_model_secs = 30000000000000000000
		self.summary_dir = None
		self.summary_frequency = 20
		self.path_to_saved_model = path_to_saved_model



