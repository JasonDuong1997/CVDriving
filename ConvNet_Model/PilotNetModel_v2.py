import tensorflow as tf

is_training = True

def weight(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1, name=name)
	return tf.Variable(initial)

def bias(shape, name):
	initial = tf.constant(0.1, shape=shape, name=name)
	return tf.Variable(initial)

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

def PilotNetV2_Model(x, WIDTH, HEIGHT, n_outputs, pool_s=2):
	W_fc_input = 8*10
	print(W_fc_input*64)

	# DEFINING WEIGHTS
	# Conv (conv): [filter_width, filter_height, channels, # of filters]
	# FC (fc)    : [size of downsampled image * size of layer input, # of neurons in layer]
	# Out (out)  : [# of outputs]
	W_conv1 = weight([5,5,  3, 24], name="W_conv1")
	W_conv2 = weight([5,5, 24, 36], name="W_conv2")
	W_conv3 = weight([5,5, 36, 48], name="W_conv3")
	W_conv4 = weight([3,3, 48, 64], name="W_conv4")
	W_conv5 = weight([3,3, 64, 64], name="W_conv5")
	W_fc1 =  weight([W_fc_input*64, 1164], name="W_fc1")
	W_fc2 =  weight([1164, 100], name="W_fc2")
	W_fc3 =  weight([100, 50], name="W_fc3")
	W_out = weight([50, n_outputs], name="W_out")
	# DEFINING BIASES
	# Conv (conv): [# number of filters]
	# FC (fc)    : [# number of filters]
	# Out (out)  : [# of outputs]
	B_conv1 = bias([24], name="B_conv1")
	B_conv2 = bias([36], name="B_conv2")
	B_conv3 = bias([48], name="B_conv3")
	B_conv4 = bias([64], name="B_conv4")
	B_conv5 = bias([64], name="B_conv5")
	B_fc1 = bias([1164], name="B_fc1")
	B_fc2 = bias([100], name="B_fc2")
	B_fc3 = bias([50], name="B_fc3")
	B_out = bias([n_outputs], name="B_out")

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
	# Output -> SoftMax	-> Steering Commands
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 3])
	print("Input Size: {}" .format(x.get_shape()))

	normalized = relu(tf.layers.batch_normalization(x, training=is_training))

	conv1 = conv2d(normalized, W_conv1, B_conv1, strides=2)
	conv1 = relu(conv1)
	conv1 = relu(tf.layers.batch_normalization(conv1, training=is_training))
	print("Conv1 Size: {}" .format(conv1.get_shape()))

	conv2 = conv2d(conv1, W_conv2, B_conv2, strides=2)
	conv2 = relu(conv2)
	conv2 = relu(tf.layers.batch_normalization(conv2, training=is_training))
	print("Conv2 Size: {}" .format(conv2.get_shape()))

	conv3 = conv2d(conv2, W_conv3, B_conv3, strides = 2)
	conv3 = relu(conv3)
	conv3 = relu(tf.layers.batch_normalization(conv3, training=is_training))
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

	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	fc2 = dropout(fc2, 0.5)

	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)
	fc3 = dropout(fc3, 0.5)

	output = tf.matmul(fc3, W_out) + B_out

	return output