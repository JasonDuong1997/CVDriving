import tensorflow as tf
import math


### VARIABLES ###
def weight(shape, n_inputs, name):
	# calculating standard deviation based on number of inputs
	std_dev = math.sqrt(2.0/n_inputs)
	# numbers chosen more than 2 std devs away are thrown away and re-picked
	initial_val = tf.truncated_normal(shape, stddev=std_dev)
	return tf.Variable(initial_val, name=name)

def bias(shape, name):
	initial_val = tf.constant(0.1, shape=shape)
	return tf.Variable(initial_val, name=name)


### ACTIVATION FUNCTIONS ###
def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.tanh(x)

def sigmoid(x):
	return tf.sigmoid(x)


### OPERATIONS ###
def conv2d(x, weight, bias, strides=1):
	return tf.nn.conv2d(x, weight, strides=[1, strides,strides, 1], padding="SAME") + bias

def max_pool2d(x, kernel=2):
	return tf.nn.max_pool(x, ksize=[1, kernel,kernel, 1], strides=[1, kernel,kernel, 1], padding="SAME")

def dropout(x, drop_rate=0.5, is_training=True):
	return tf.layers.dropout(x, rate=drop_rate, training=is_training)


### MODEL ###
def PilotNetV2_Model(x, WIDTH, HEIGHT, n_outputs, pool_s=2, is_training=True):
	W_conv_input = WIDTH*HEIGHT*3
	W_fc_input = 8*10

	# DEFINING WEIGHTS
	# Convolution (conv):   [filter_width, filter_height, channels, # of filters]
	# Fully-Connected (fc): [size of downsampled image * size of layer input, # of neurons in layer]
	# Output (out): 		[# of outputs]
	W_conv1 = weight([5,5,  3, 24], n_inputs=W_conv_input, name="W_conv1")
	W_conv2 = weight([5,5, 24, 36], n_inputs=3*24, name="W_conv2")
	W_conv3 = weight([5,5, 36, 48], n_inputs=24*35, name="W_conv3")
	W_conv4 = weight([3,3, 48, 64], n_inputs=36*48, name="W_conv4")
	W_conv5 = weight([3,3, 64, 64], n_inputs=48*64, name="W_conv5")
	W_fc1   = weight([W_fc_input*64, 1164], n_inputs=64*64, name="W_fc1")
	W_fc2   = weight([1164, 100],           n_inputs=1164, name="W_fc2")
	W_fc3   = weight([100, 50],             n_inputs=100, name="W_fc3")
	W_fc4   = weight([50, 10],              n_inputs=50, name="W_fc4")
	W_out   = weight([10, n_outputs],       n_inputs=10, name="W_out")
	# DEFINING BIASES
	# Convolution (conv): 	[# number of filters]
	# Fully-Connected (fc): [# number of filters]
	# Output (out): 		[# of outputs]
	B_conv1 = bias([24],   name="B_conv1")
	B_conv2 = bias([36],   name="B_conv2")
	B_conv3 = bias([48],   name="B_conv3")
	B_conv4 = bias([64],   name="B_conv4")
	B_conv5 = bias([64],   name="B_conv5")
	B_fc1   = bias([1164], name="B_fc1")
	B_fc2   = bias([100],  name="B_fc2")
	B_fc3   = bias([50],   name="B_fc3")
	B_fc4   = bias([10],   name="B_fc4")
	B_out   = bias([n_outputs], name="B_out")

	# DEFINING PilotNet ARCHITECTURE
	# Input Image(width = 80, height = 60, RGB) ->
	# Normalization ->
	# Convolution(5x5) -> Relu -> Normalization -> Relu ->
	# Convolution(5x5) -> Relu -> Normalization -> Relu ->
	# Convolution(5x5) -> Relu -> Normalization -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Fully Connected Layer(1164) -> Relu -> Dropout
	# Fully Connected Layer(100) -> Relu -> Dropout
	# Fully Connected Layer(50) -> Relu ->
	# Output -> Steering Angle
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 3])
	print("Input Size: {}" .format(x.get_shape()))

	# to normalize in hidden layers, add the normalization layer:
	# 1. right after fc or conv layers
	# 2. right before non-linearities
	normalized = tf.layers.batch_normalization(x, training=is_training, trainable=True)

	conv1 = conv2d(normalized, W_conv1, B_conv1, strides=2)
	conv1 = tanh(conv1)
	# conv1 = tanh(tf.layers.batch_normalization(conv1, training=is_training, trainable=True))

	conv2 = conv2d(conv1, W_conv2, B_conv2, strides=2)
	conv2 = tanh(conv2)
	# conv2 = tanh(tf.layers.batch_normalization(conv2, training=is_training, trainable=True))

	conv3 = conv2d(conv2, W_conv3, B_conv3, strides=2)
	conv3 = tanh(conv3)
	# conv3 = tanh(tf.layers.batch_normalization(conv3, training=is_training, trainable=True))

	conv4 = conv2d(conv3, W_conv4, B_conv4, strides=1)
	conv4 = tanh(conv4)
	# conv4 = tanh(tf.layers.batch_normalization(conv4, training=is_training, trainable=True))

	conv5 = conv2d(conv4, W_conv5, B_conv5, strides=1)
	conv5 = tanh(conv5)
	# conv5 = tanh(tf.layers.batch_normalization(conv5, training=is_training, trainable=True))

	# flatten to 1 dimension for fully connected layers
	flat_img = tf.reshape(conv5, shape=[-1, W_fc1.get_shape().as_list()[0]])

	fc1 = tanh(tf.matmul(flat_img, W_fc1) + B_fc1)
	# fc1 = dropout(fc1, 0.2, is_training)

	fc2 = tanh(tf.matmul(fc1, W_fc2) + B_fc2)
	# fc2 = dropout(fc2, 0.3, is_training)

	fc3 = tanh(tf.matmul(fc2, W_fc3) + B_fc3)
	# fc3 = dropout(fc3, 0.3, is_training)

	fc4 = tanh(tf.matmul(fc3, W_fc4) + B_fc4)

	# pure linear output
	output = tf.matmul(fc4, W_out) + B_out

	return output
