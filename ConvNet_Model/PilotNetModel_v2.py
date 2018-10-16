import tensorflow as tf
import math


### TF VARIABLES ###
def weight(shape, n_inputs, name):
	# calculating standard deviation based on number of inputs
	std_dev = math.sqrt(2.0/n_inputs)
	# numbers chosen more than 2 std devs away are thrown away and re-picked
	initial_val = tf.truncated_normal(shape, stddev=std_dev)
	return tf.Variable(initial_val, name=name)

def bias(shape, name):
	initial_val = tf.constant(0.1, shape=shape)
	return tf.Variable(initial_val, name=name)


### ACTIVATION/TRANSFER FUNCTIONS ###
def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.tanh(x)

def sigmoid(x):
	return tf.sigmoid(x)


### OPERATIONS ###
def conv2d(x, weight, bias, strides=1):
	return tf.nn.conv2d(x, weight, strides=[1, strides,strides, 1], padding="SAME") + bias

def max_pool2d(x, strides=2):
	return tf.nn.max_pool(x, ksize=[1, strides,strides, 1], strides=[1, strides,strides, 1], padding="SAME")

def dropout(x, drop_rate=0.5, is_training=True):
	return tf.layers.dropout(x, rate=drop_rate, training=is_training)

def normalize(x, is_training):
	return tf.layers.batch_normalization(x, training=is_training, trainable=True)


### MODEL DEFINITION ###
def cnn_model(x, WIDTH, HEIGHT, n_outputs, is_training):
	W_conv_input = WIDTH*HEIGHT*3
	W_fc_input = 8*10

	# DEFINING WEIGHTS
	# Convolution (conv):   [filter_width, filter_height, channels, # of filters]
	# Fully-Connected (fc): [# of neurons in input layer, # of neurons to output]
	# Output (out): 		[# of model outputs]
	W_conv1 = weight([7,7,  3, 24], n_inputs=W_conv_input, 	name="W_conv1")
	W_conv2 = weight([7,7, 24, 48], n_inputs=24*48, 		name="W_conv2")
	W_conv3 = weight([7,7, 48, 36], n_inputs=48*36, 		name="W_conv3")
	W_conv4 = weight([5,5, 36, 48], n_inputs=36*48, 		name="W_conv4")
	W_conv5 = weight([5,5, 48, 96], n_inputs=48*96, 		name="W_conv5")
	W_conv6 = weight([5,5, 96, 96], n_inputs=96*96, 		name="W_conv6")
	W_conv7 = weight([3,3, 96, 64], n_inputs=96*64, 		name="W_conv7")
	W_conv8 = weight([3,3, 64, 64], n_inputs=64*64, 		name="W_conv8")
	W_conv9 = weight([3,3, 64, 128], n_inputs=64*64, 		name="W_conv9")
	W_fc1   = weight([W_fc_input*128, 720], n_inputs=40*40, name="W_fc1")
	W_fc2   = weight([720, 360],           	n_inputs=466,	name="W_fc2")
	W_fc3   = weight([360, 180],            n_inputs=233, 	name="W_fc3")
	W_out   = weight([180, n_outputs],      n_inputs=12, 	name="W_out")
	# DEFINING BIASES
	# Convolution (conv): 	[# number of filters]
	# Fully-Connected (fc): [# number of filters]
	# Output (out): 		[# of outputs]
	B_conv1 = bias([24],   		name="B_conv1")
	B_conv2 = bias([48],   		name="B_conv2")
	B_conv3 = bias([36],   		name="B_conv3")
	B_conv4 = bias([48],   		name="B_conv4")
	B_conv5 = bias([96],   		name="B_conv5")
	B_conv6 = bias([96],   		name="B_conv6")
	B_conv7 = bias([64],   		name="B_conv7")
	B_conv8 = bias([64],   		name="B_conv8")
	B_conv9 = bias([128],   	name="B_conv9")
	B_fc1   = bias([720], 		name="B_fc1")
	B_fc2   = bias([360],  		name="B_fc2")
	B_fc3   = bias([180],   	name="B_fc3")
	B_out   = bias([n_outputs], name="B_out")

	# DEFINING MODEL ARCHITECTURE
	# Input Image(width = 80, height = 60, RGB) ->
	# Normalization ->
	# Convolution(7x7) -> Relu ->
	# Convolution(7x7) -> Relu ->
	# Convolution(7x7) -> Relu -> Normalize -> MaxPool -> Dropout
	# Convolution(5x5) -> Relu ->
	# Convolution(5x5) -> Relu ->
	# Convolution(5x5) -> Relu -> Normalize -> MaxPool -> Dropout
	# Convolution(3x3) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Fully Connected Layer(720) -> Relu ->
	# Fully Connected Layer(360) -> Relu ->
	# Fully Connected Layer(180) -> Relu ->
	# Output -> Steering Angle
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 3])
	print("Input Size: {}" .format(x.get_shape()))

	# to normalize in hidden layers, add the normalization layer:
	# 1. right after fc or conv layers
	# 2. right before non-linearities
	normalized = tf.layers.batch_normalization(x, training=is_training, trainable=True)

	conv1 = conv2d(normalized, W_conv1, B_conv1, strides=2)
	conv1 = relu(conv1)

	conv2 = conv2d(conv1, W_conv2, B_conv2, strides=1)
	conv2 = relu(conv2)

	conv3 = conv2d(conv2, W_conv3, B_conv3, strides=1)
	conv3 = relu(conv3)
	conv3 = normalize(conv3, is_training)
	conv3 = max_pool2d(conv3, strides=2)
	conv3 = dropout(conv3, 0.5, is_training)

	conv4 = conv2d(conv3, W_conv4, B_conv4, strides=1)
	conv4 = relu(conv4)

	conv5 = conv2d(conv4, W_conv5, B_conv5, strides=1)
	conv5 = relu(conv5)

	conv6 = conv2d(conv5, W_conv6, B_conv6, strides=1)
	conv6 = relu(conv6)
	conv6 = normalize(conv6, is_training)
	conv6 = max_pool2d(conv6, strides=2)
	conv6 = dropout(conv6, 0.5, is_training)

	conv7 = conv2d(conv6, W_conv7, B_conv7, strides=1)
	conv7 = relu(conv7)

	conv8 = conv2d(conv7, W_conv8, B_conv8, strides=1)
	conv8 = relu(conv8)

	conv9 = conv2d(conv8, W_conv9, B_conv9, strides=1)
	conv9 = relu(conv9)

	# flatten to 1 dimension for fully connected layers
	flat_img = tf.reshape(conv9, shape=[-1, W_fc1.get_shape().as_list()[0]])

	fc1 = relu(tf.matmul(flat_img, W_fc1) + B_fc1)
	fc1 = dropout(fc1, 0.5)

	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	fc2 = dropout(fc2, 0.5)

	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)

	# pure linear output
	output = tf.matmul(fc3, W_out) + B_out

	return output
