import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

def encoder_module(self):
	'''
	Encoding of the question is done using the simple BiGRU encoding
	x = (x[1], x[2], ..., x[n])
	gives 2 outputs
	1) the forward sequence h_f
	2) the backward sequence h_b
	these two are concatenated to give the output of encoder module as 
	h[i] = [h_f[i]; h_b[i]]
	'''
	encoder_cell = GRUCell(self.n_hidden) # we don't need two seperate cells for forward and backwards reading
	# op is a tuple : (output_fw, output_bw)
	(encoder_fw_outputs, encoder_bw_outputs), encoder_states = \
		tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
		cell_bw = encoder_cell, inputs = ,sequence_length = ,
		time_major = True)

	# getting the final encoder_outputs
	encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), axis = 2)

	# encoder_state[0] : both FW/FW
	# encoder_state[1] : both BW/BW
	encoder_final_state_c = tf.concat((encoder_states[0].c, encoder_states[1].c), axis = 1)
	encoder_final_state_h = tf.concat((encoder_states[0].h, encoder_states[1].h), axis = 1)
	encoder_final_state = LSTMStateTuple(c = encoder_final_state_c, h = encoder_final_state_h)

	# we have obtained the final state
	# encoder_final_state is the output sequence of the encoder module (h)
	return encoder_final_state

def decoder_module(self):
	'''
	This is the basic decoding mathematics given in the paper
	w --> previous word embedding
	c --> context vector
	s --> decoder state
	r --> readout state

	s[t] = GRU(w[t-1], c[t-1], c[t-1])
	s[0] = tanh(W_d*h1_back + bias)
	e[t][i] = v_a' * tanh(W_a*s[t-1] + U_a*h[i])
	alpha[t][i] = softmax(e[t][i])
	c[t] = sum_over_i[1,n] alpha[t][i] * h[i]

	r[t] = W_r*w[t-1] + U_r*c[t] + V_r*s[t]
	m[t] = [max{r[t][2j-1], r[t][2j]}]' for j=1,...,d
	p(y[t]|y[1],....,y[t-1]) = softmax(W_o*m[t])
	'''
	encoder_final_state = encoder_module()

	# Declaring the weights and biases
	W_dec = tf.Variable(tf.truncated_normal(shape = []), name = 'W_dec')
	b_dec = tf.Variable(tf.truncated_normal(shape = []), name = 'b_dec')
	W_act = tf.Variable(tf.truncated_normal(shape = []), name = 'W_act')
	U_act = tf.Variable(tf.truncated_normal(shape = []), name = 'U_act')
	v_act = tf.Variable(tf.truncated_normal(shape = []), name = 'v_act')
	W_ro = tf.Variable(tf.truncated_normal(shape = []), name = 'W_ro')
	U_ro = tf.Variable(tf.truncated_normal(shape = []), name = 'U_ro')
	V_ro = tf.Variable(tf.truncated_normal(shape = []), name = 'V_ro')
	W_o = tf.Variable(tf.truncated_normal(shape = []), name = 'W_o')

	# Decoder Cell
	decoder_cell = GRUCell(n_hidden)

	# mathematical equations
	decoder_state_init = tf.tanh(tf.add(tf.matmul(input_state, W_dec), b_dec))
	decoder_op, decoder_state = tf.nn.dynamic_rnn(decoder_cell, previous_state)
	e = tf.transpose(v_act) * tf.tanh(tf.matmul(decoder_state, W_act) + tf.matmul(encoder_state_final, U_act))














