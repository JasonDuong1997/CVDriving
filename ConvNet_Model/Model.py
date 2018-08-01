import tensorflow as tf
import tflearn

# Process
"""
input -> (convolution -> max pooling) ->  fully connected layer -> output layer
"""

def conv2d(x, weight, bias, strides=1):
	conv = tf.nn.conv2d(x, weight, strides=[1, strides,strides, 1], padding="SAME")
	conv = tf.nn.bias_add(conv, bias)
	return conv

def relu(x):
	return tf.nn.relu(x)

def max_pool2d(x, kernel=2):
	return tf.nn.max_pool(x, ksize=[1, kernel,kernel, 1], strides=[1, kernel,kernel, 1], padding="SAME")


def ConvNN_Model(x, WIDTH, HEIGHT, n_outputs, pool_s=2):
	W_fc_input = 16*20
	print(W_fc_input*64)
	# DEFINING WEIGHTS
	# Conv: [filter_width, filter_height, channels, # of filters]
	# FC:   [size of downsampled image * size of layer input, # of neurons in layer]
	# Out:  [# of outputs]
	W_conv1 = tf.Variable(tf.random_normal([5,5, 1, 32]), name="W_conv1")
	W_conv2 = tf.Variable(tf.random_normal([5,5, 32, 64]), name="W_conv2")
	W_fc =  tf.Variable(tf.random_normal([W_fc_input*64, 1024]), name="W_fc")
	W_out = tf.Variable(tf.random_normal([1024, n_outputs]), name="W_out")
	# DEFINING BIASES
	# Conv: [# number of filters]
	# FC:   [# number of filters]
	# Out:  [# of outputs]
	B_conv1 = tf.Variable(tf.random_normal([32]), name="B_conv1")
	B_conv2 = tf.Variable(tf.random_normal([64]), name="B_conv2")
	B_fc = tf.Variable(tf.random_normal([1024]), name="B_fc")
	B_out = tf.Variable(tf.random_normal([n_outputs]), name="B_out")

	# DEFINING ARCHITECTURE
	# Input ->
	# Convolution -> Relu -> MaxPooling ->
	# Convolution -> Relu -> MaxPooling ->
	# Fully Connected Layer ->
	# Output
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 1])
	print("Input: {}" .format(x.get_shape()))
	conv1 = conv2d(x, W_conv1, B_conv1, strides=1)
	conv1 = relu(conv1)
	print("Conv1: {}" .format(conv1.get_shape()))
	conv1 = max_pool2d(conv1, kernel=pool_s)
	print("MPool1: {}" .format(conv1.get_shape()))

	conv2 = conv2d(conv1, W_conv2, B_conv2, strides=1)
	conv2 = relu(conv2)
	conv2 = max_pool2d(conv2, kernel=pool_s)
	print("MPool2: {}" .format(conv2.get_shape()))

	fc = tf.reshape(conv2, shape=[-1, W_fc.get_shape().as_list()[0]])
	fc = relu(tf.matmul(fc, W_fc) + B_fc)

	output = tf.matmul(fc, W_out) + B_out

	return output