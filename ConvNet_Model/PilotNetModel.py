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

def dropout(x, drop_rate=0.5):
	return tf.layers.dropout(x, rate=drop_rate)

def PilotNet_Model(x, WIDTH, HEIGHT, n_outputs, pool_s=2):
	W_fc_input = 8*10
	print(W_fc_input*64)
	# DEFINING WEIGHTS
	# Conv (conv): [filter_width, filter_height, channels, # of filters]
	# FC (fc)    : [size of downsampled image * size of layer input, # of neurons in layer]
	# Out (out)  : [# of outputs]
	W_conv1 = tf.Variable(tf.random_normal([5,5,  3, 24]), name="W_conv1")
	W_conv2 = tf.Variable(tf.random_normal([5,5, 24, 36]), name="W_conv2")
	W_conv3 = tf.Variable(tf.random_normal([5,5, 36, 48]), name="W_conv3")
	W_conv4 = tf.Variable(tf.random_normal([3,3, 48, 64]), name="W_conv4")
	W_conv5 = tf.Variable(tf.random_normal([3,3, 64, 64]), name="W_conv5")
	W_fc1 =  tf.Variable(tf.random_normal([W_fc_input*64, 1164]), name="W_fc1")
	W_fc2 =  tf.Variable(tf.random_normal([1164, 100]), name="W_fc2")
	W_fc3 =  tf.Variable(tf.random_normal([100, 50]), name="W_fc3")
	W_out = tf.Variable(tf.random_normal([50, n_outputs]), name="W_out")
	# DEFINING BIASES
	# Conv (conv): [# number of filters]
	# FC (fc)    : [# number of filters]
	# Out (out)  : [# of outputs]
	B_conv1 = tf.Variable(tf.random_normal([24]), name="B_conv1")
	B_conv2 = tf.Variable(tf.random_normal([36]), name="B_conv2")
	B_conv3 = tf.Variable(tf.random_normal([48]), name="B_conv3")
	B_conv4 = tf.Variable(tf.random_normal([64]), name="B_conv4")
	B_conv5 = tf.Variable(tf.random_normal([64]), name="B_conv5")
	B_fc1 = tf.Variable(tf.random_normal([1164]), name="B_fc1")
	B_fc2 = tf.Variable(tf.random_normal([100]), name="B_fc2")
	B_fc3 = tf.Variable(tf.random_normal([50]), name="B_fc3")
	B_out = tf.Variable(tf.random_normal([n_outputs]), name="B_out")

	# DEFINING PilotNet ARCHITECTURE
	# Input Image(width = 80, height = 62, YUV) ->
	# Normalization ->
	# Convolution(5x5) -> Relu ->
	# Convolution(5x5) -> Relu ->
	# Convolution(5x5) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Fully Connected Layer(1164) ->
	# Fully Connected Layer(100) ->
	# Fully Connected Layer(50) ->
	# Output -> Steering Commands
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 3])
	print("Input Size: {}" .format(x.get_shape()))

	conv1 = conv2d(x, W_conv1, B_conv1, strides=2)
	conv1 = relu(conv1)
	print("Conv1 Size: {}" .format(conv1.get_shape()))

	conv2 = conv2d(conv1, W_conv2, B_conv2, strides=2)
	conv2 = relu(conv2)
	print("Conv2 Size: {}" .format(conv2.get_shape()))

	conv3 = conv2d(conv2, W_conv3, B_conv3, strides = 2)
	conv3 = relu(conv3)
	print("Conv3 Size: {}" .format(conv3.get_shape()))

	conv4 = conv2d(conv3, W_conv4, B_conv4, strides = 1)
	conv4 = relu(conv4)
	print("Conv4 Size: {}" .format(conv4.get_shape()))

	conv5 = conv2d(conv4, W_conv5, B_conv5, strides = 1)
	conv5 = relu(conv5)
	print("Conv5 Size: {}" .format(conv5.get_shape()))

	# flatten to 1 dimension for fully connected layers
	flat_img = tf.reshape(conv5, shape=[-1, W_fc1.get_shape().as_list()[0]])
	print("Reshape Size: {}" .format(flat_img.get_shape()))
	fc1 = relu(tf.matmul(flat_img, W_fc1) + B_fc1)
	fc1 = dropout(fc1, 0.5)
	print("FC1 Size: {}" .format(fc1.get_shape()))
	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	fc2 = dropout(fc2, 0.3)
	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)

	output = tf.matmul(fc3, W_out) + B_out

	return output