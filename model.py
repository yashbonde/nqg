# This is the model file for the NQG

# importing the dependencies
import tensorflow as tf
import numpy as np
import os
from six.moves import xrange

class NQG(object):
	"""docstring for NQG"""
	def __init__(self, args, sess):
		# First defining the tuanble hyper paratmeters
		self.batch_size = args.batch_size
		self.n_epochs = args.n_epochs

		# defning the session
		self.sess = sess

		# defining tensorflow ops
		self.input_x = tf.placeholder(tf.float32, shape = [])

		# defining the logs
		self.acc_log = []
		self.loss_log = []

		# defining misc.
		self.test_true = args.test_true
		self.checkpoint_dir = config.checkpoint_dir # the checkpoint directory to load the model
		if not os.path.isdir(self.checkpoint_dir):
			raise Exception("[!] Directory {0} not found".format(self.checkpoint_dir))

	def _encoder_module(self):
		# This module makes the encoder
		'''
		input(x) = (x[1], x[2], ... ,x[n])
			from the above input we calculate the bi-directional hidden states
		hidden_fw = (h_fw[1], h_fw[2], ... ,h_fw[n])
		hidden_bw = (h_bw[1], h_bw[2], ... ,h_bw[n])
			from these two we determine the final hidden state as
		hidden[i] = hidden_fw[i] + hidden_bw[i]
			basically concatenating or extending the list
		'''
		# We have to use a BiGRU
		cell_fw = rnn.GRUCell()
		cell_bw = rnn.GRUCell()

		pass

	def _decoder_module(self):
		# This module makes the decoder
		'''
		decoder_state[t] = GRU(w[t-1], c[t-1], decoder_state[t-1]) --> Decoder State
		decoder_state[0] = tanh((W_d * h_back[0]) + b) --> Decoder State at time = 0
		e[t][i] = transpose(v_d) * tanh((W_a * decoder_state[t -1]) + (U_a * hidden[i])
		aplha[i] = softmax(e[t][i])
		ccs[t] = sum_over_i(aplha[t][i] * hidden[i]) --> Current context state
		r[t] = (W_r * w[t-1]) + (U_r * ccs[t]) + (V_r * decoder_state[t]) --> Read Out state
		m[t] = transpose(max(r[t][2j-1], r[t][2j])) --> Max out state
		p(y[t]| y[0], ... ,y[t-1]) = softmax(W_op * m[t]) --> final word prediction
			In the last line we basically are saying that to predict the next word provided that we have been
			given all the previous words, basically calculating the conditional probability, is determined
			by a softmax function of final weight matrix and max_out_state
		'''
		# making the Variable matrix
		W_dec = tf.Variable(tf.float32, shape = [], name = 'W_dec') # --> decoder weight matrix
		b_dec = tf.Variable(tf.float32, shape = [], name = 'b_dec') # --> decoder bias vector
		W_act = tf.Variable(tf.float32, shape = [], name = 'W_act') # --> activation weight matrix #1 (used with decoder_state)
		U_act = tf.Variable(tf.float32, shape = [], name = 'U_act') # --> activation weight matrix #2 (used with hidden_state)
		W_ro = tf.Variable(tf.float32, shape = [], name = 'W_ro') # --> readout state matrix #1 
		U_ro = tf.Variable(tf.float32, shape = [], name = 'U_ro') # --> readout state matrix #2
		V_ro = tf.Variable(tf.float32, shape = [], name = 'V_ro') # --> readout state matrix #3
		W_op = tf.Variable(tf.float32, shape = [], name = 'W_op') # --> Final output weight matrix

		pass

	def _build(self):
		# This module makes the model
		self._encoder_module()
		self._decoder_module()
		pass

	def train(self, data):
		# This is the training module
		pass

	def test(self, data, name = 'Validation'):
		# This model is used for testing/validation
		pass

	def run(self, train_data, test_data):
		if not self.test_true:
			# This is used when the model is being used for training
			pass
		else:
			# This part is used when model is being used for testing
			self.load_model()

			test_loss = np.sum(self.test(test_data, label='Test')[0])
			test_acc = self.test(test_data)

			state = {
				'test_loss': test_loss,
				'accuracy': test_acc
			}
			
			print(state)

	def load_model(self):
		print("[*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path: 
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception("[!] Test mode but no checkpoint found")
