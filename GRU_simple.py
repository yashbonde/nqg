# This is the simple GRU with tensorflow for practice
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding

# STEP 1: GENERATION OF DATA
data_dim = 128
max_len = 21
num_classes = 10
batch_size = 320
state_size = 16
init_size = 64

# Generate dummy training data
vocab_size = 1771
x_train = np.random.randint(vocab_size, size = (batch_size, max_len)) # This is a sample data for any text file

# where the vocab_size = 1771, number of sentences = 320, length  of each sentence - 21
# Embedding teh data using keras, if this wirs we will use this only
model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim = data_dim, input_length = max_len))
model.compile('rmsprop', 'mse')
x_train = model.predict(x_train)
y_train = np.random.randint(2, size = (batch_size, num_classes))

'''# Generate dummy validation data
x_val = np.random.random((batch_size * 3, max_len, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))'''

# STEP 2: MAKING THE MODEL
x = tf.placeholder(tf.float32, [None, max_len, data_dim], name = 'x') # (320, 21, 128)
y_ = tf.placeholder(tf.float32, [None, num_classes], name = 'y_') # (320, 10)

cell = rnn.GRUCell(num_units = state_size) # state size = 16
# We let the function have it's own initialisation state
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
# print(rnn_outputs) # -> (?, 21, 16)
# print(final_state) # -> (?, 16)

rnn_outputs = tf.reshape(rnn_outputs, [batch_size, 16*21], name = 'reshaped')
# print(rnn_outputs) # -> (batch_size, 336)

W = tf.Variable(tf.truncated_normal(shape = [16*21, num_classes], stddev = 0.1), name = 'W')
b = tf.Variable(tf.truncated_normal(shape = [num_classes], stddev = 0.1), name = 'b')
y = tf.add(tf.matmul(rnn_outputs, W), b, name = 'y')

# Training and running of model
loss = tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_)
train_step = tf.train.AdamOptimizer().minimize(loss)
loss_val = tf.reduce_mean(loss)

# Defining the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
	loss_eval = loss_val.eval(feed_dict = {x: x_train, y_: y_train}, session = sess)
	print("Step {0}, loss = {1}".format(i, loss_eval))
	sess.run([train_step], feed_dict = {x: x_train, y_: y_train})
