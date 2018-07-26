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
	W_fc_input = int(WIDTH*HEIGHT/pow(pool_s, 2))

	# DEFINING WEIGHTS & BIASES
	# Conv: [filter_width, filter_height, channels, # of filters]
	# FC:   [size of downsampled image * size of layer input, # of neurons in layer]
	# Out:  [# of outputs]
	weights = {"W_conv1": tf.Variable(tf.random_normal([5,5, 1, 64])),
			   "W_conv2": tf.Variable(tf.random_normal([5,5, 64, 128])),
			   "W_fc": tf.Variable(tf.random_normal([W_fc_input*128, 1024])),
			   "W_out": tf.Variable(tf.random_normal([1024, n_outputs]))}
	# Conv: [# number of filters]
	# FC:   [# number of filters]
	# Out:  [# of outputs]
	biases = {"B_conv1": tf.Variable(tf.random_normal([64])),
			  "B_conv2": tf.Variable(tf.random_normal([128])),
			  "B_fc": tf.Variable(tf.random_normal([1024])),
			  "B_out": tf.Variable(tf.random_normal([n_outputs]))}

	# DEFINING ARCHITECTURE
	# Input ->
	# Convolution -> Relu -> MaxPooling ->
	# Convolution -> Relu -> MaxPooling ->
	# Fully Connected Layer ->
	# Output
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 1])
	conv1 = conv2d(x, weights["W_conv1"], biases["B_conv1"], strides=1)
	conv1 = relu(conv1)
	conv1 = max_pool2d(conv1, kernel=pool_s)

	conv2 = conv2d(conv1, weights["W_conv2"], biases["B_conv2"], strides=1)
	conv2 = relu(conv2)
	conv2 = max_pool2d(conv2, kernel=pool_s)


	fc = tf.reshape(conv2, [-1, weights["W_fc"].get_shape().as_list()[0]])
	fc = relu(tf.matmul(fc, weights["W_fc"]) + biases["B_fc"])

	output = tf.matmul(fc, weights["W_out"]) + biases["B_out"]

	return output