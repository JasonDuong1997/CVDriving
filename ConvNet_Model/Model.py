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

def dropout(x, keep_rate=0.5):
	return tf.layers.dropout(x, rate=keep_rate)

def ConvNN_Model(x, WIDTH, HEIGHT, n_outputs, pool_s=2):
	W_fc_input = 8*10
	print(W_fc_input*64)
	# DEFINING WEIGHTS
	# Conv: [filter_width, filter_height, channels, # of filters]
	# FC:   [size of downsampled image * size of layer input, # of neurons in layer]
	# Out:  [# of outputs]
	W_conv1 = tf.Variable(tf.random_normal([5,5,  1, 24]), name="W_conv1")
	W_conv2 = tf.Variable(tf.random_normal([5,5, 16, 32]), name="W_conv2")
	W_conv3 = tf.Variable(tf.random_normal([5,5, 32, 48]), name="W_conv3")
	W_conv4 = tf.Variable(tf.random_normal([3,3, 48, 64]), name="W_conv4")
	W_conv5 = tf.Variable(tf.random_normal([3,3, 64, 64]), name="W_conv5")
	W_fc1 =  tf.Variable(tf.random_normal([W_fc_input*64, 1024]), name="W_fc1")
	W_fc2 =  tf.Variable(tf.random_normal([1024, 100]), name="W_fc2")
	W_fc3 =  tf.Variable(tf.random_normal([100, 50]), name="W_fc3")
	W_out = tf.Variable(tf.random_normal([50, n_outputs]), name="W_out")
	# DEFINING BIASES
	# Conv: [# number of filters]
	# FC:   [# number of filters]
	# Out:  [# of outputs]
	B_conv1 = tf.Variable(tf.random_normal([24]), name="B_conv1")
	B_conv2 = tf.Variable(tf.random_normal([32]), name="B_conv2")
	B_conv3 = tf.Variable(tf.random_normal([48]), name="B_conv3")
	B_conv4 = tf.Variable(tf.random_normal([64]), name="B_conv4")
	B_conv5 = tf.Variable(tf.random_normal([64]), name="B_conv5")
	B_fc1 = tf.Variable(tf.random_normal([1024]), name="B_fc1")
	B_fc2 = tf.Variable(tf.random_normal([100]), name="B_fc2")
	B_fc3 = tf.Variable(tf.random_normal([50]), name="B_fc3")
	B_out = tf.Variable(tf.random_normal([n_outputs]), name="B_out")

	# DEFINING ARCHITECTURE
	# Input Image(width = 80, height = 62) ->
	# Convolution(5x5) -> Relu -> MaxPooling ->
	# Convolution(5x5) -> Relu -> MaxPooling ->
	# Convolution(5x5) -> Relu ->
	# Convolution(3x3) -> Relu -> MaxPooling ->
	# Convolution(3x3) -> Relu ->
	# Fully Connected Layer(1024) ->
	# Fully Connected Layer(100) ->
	# Fully Connected Layer(50) ->
	# Output -> Steering Commands
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

	conv3 = conv2d(conv2, W_conv3, B_conv3, strides = 1)
	conv3 = relu(conv3)

	conv4 = conv2d(conv3, W_conv4, B_conv4, strides = 1)
	conv4 = relu(conv4)
	conv4 = max_pool2d(conv4, kernel = pool_s)

	conv5 = conv2d(conv4, W_conv5, B_conv5, strides = 1)
	conv5 = relu(conv5)

	# flatten to 1 dimension for fully connected layers
	flat_img = tf.reshape(conv5, shape=[-1, W_fc1.get_shape().as_list()[0]])
	fc1 = relu(tf.matmul(flat_img, W_fc1) + B_fc1)
	fc1 = dropout(fc1, 0.5)
	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	fc2 = dropout(fc2, 0.5)
	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)

	output = tf.matmul(fc3, W_out) + B_out

	return output