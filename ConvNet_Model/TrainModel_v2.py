import tensorflow as tf
from ConvNet_Model.PilotNetModel_v2 import  PilotNetV2_Model
import matplotlib.pyplot as plt
import numpy as np
import time

### HELPER FUNCTIONS ###
def round_decimal(num, n_positions):
	format = "%." + str(n_positions) + "f"
	return float(format % num)


def k_fold_splitter(training_data, k=5):	# splitting training data into k equal parts
	print("BEGIN: K-Fold Split with K={}" .format(k))
	fold_len = int(len(training_data)/k)

	split_data = []
	for i in range(k):
		start_index = i*fold_len
		split_data.append(training_data[start_index:(start_index + fold_len - 1)])

	print("Data Kept: {} %" .format(fold_len*k/len(training_data*100)))
	print("END: K-Fold Split")
	return split_data

def k_fold_selector(split_data, index):	# sectioning out train and test set
	print("BEGIN: K-Fold Selector with Index={}" .format(index))
	test_data = split_data[index]

	train_data = []
	for i in range(len(split_data)):
		if (i != index):
			for item in split_data[i]:
				train_data.append(item)

	train_x = [i[0] for i in train_data]
	train_y = [i[1] for i in train_data]
	test_x 	= [i[0] for i in test_data]
	test_y 	= [i[1] for i in test_data]
	print("END: K-Fold Selector")
	return train_x, train_y, test_x, test_y

def early_stop(loss_check, strikes, threshold, loss_monitor="validation"):	# checks the condition on when to finish training
	if (loss_monitor == "validation"):
		delta_loss = loss_check[1] - loss_check[0]
		# if the change in loss is less than [tolerance]% or if the loss increased, add a strike
		if (abs(delta_loss) < abs(threshold*loss_check[0]) or delta_loss > loss_check[0]):
			strikes += 1
		else:
			strikes = 0
	return strikes


def cyclical_lr(epoch, amplitude, period):	#todo make this work with decay LR
	return np.sin(epoch*2*np.pi/period)*amplitude


### GLOBAL VARIABLES ###
loaded_data = np.load("udacity_trainingData_processed.npy")

# separating out the data into training and validation set
# x_set = [i[0] for i in loaded_data]
# train_x = x_set[:-test_size]
# test_x = x_set[-test_size:]
# y_set = [i[1] for i in loaded_data]
# train_y = y_set[:-test_size]
# test_y = y_set[-test_size:]
k_splits = 10
k_split_data = k_fold_splitter(loaded_data, k_splits)
train_x, train_y, test_x, test_y = k_fold_selector(k_split_data, 9)
print("train/test X: {}, {}".format(len(train_x), len(test_x)))
print("train/test Y: {}, {}".format(len(train_y), len(test_y)))

# training variables
#test_size = int(len(loaded_data)*0.10)
batch_size = 64  	# number of images per cycle (in the power of 2 because # of physical processors is similar)
n_epochs = 3000 	# number of epochs
n_outputs = 1	  	# number of outputs

# setting decaying learning rate decided using Grid Search
# *********************************** #
# Learning Rate List with 300 Epochs  #
# *********************************** #
# [LR]		[Train_Loss]	[Val_Loss]
# 3.0e-4	1.556			0.222
# 2.0e-4	1.165			0.183
# 1.5e-4	0.885			0.129
# 1.0e-4	0.680			0.107
# 9.5e-5	0.663			0.162
# 8.0e-5	0.545			0.107
# 6.0e-5	0.753			0.133
# 1.0e-5	1.226			0.214
# 1.0e-6	10.87			1.369
# ********************************** #
# optimizer variables
global_step = tf.Variable(0, trainable=False, name="global_step")
initial_learning_rate = 9e-5
epsilon = 5e-6
steps_per_epoch = len(train_x)/batch_size
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 200*steps_per_epoch, 0.90, staircase=True, name="LR_Decaying")  #todo check if this decay rate makes sense

# image input dimensions
WIDTH = 80
HEIGHT = 60
# input placeholder tensors
x = tf.placeholder("float", [None, HEIGHT, WIDTH, 3])
y = tf.placeholder("float", [None, n_outputs])

# model version
PNN_VERSION = "1.0"


### TRAINING FUNCTION ###
def ConvNN_Train(x):
	# declaring operations
	prediction = PilotNetV2_Model(x, WIDTH, HEIGHT, n_outputs, is_training=True)
	training_variables = tf.trainable_variables()	# getting list of trainable variables defined in the model
	cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y))) + tf.add_n([tf.nn.l2_loss(variable) for variable in training_variables if "B" not in variable.name])*learning_rate

	# creating optimizer with normalization
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(cost, global_step=global_step)

	# creating plot to graph the loss
	x_max = n_epochs
	x_scale = n_epochs/20
	y_max = 30
	y_scale = 1
	plt.figure(figsize=(15,8))
	plt.axis([0, x_max, 0, y_max])
	plt.grid(True)
	plt.xticks(np.arange(0, x_max, x_scale))
	plt.yticks(np.arange(0, y_max, y_scale))
	plt.xlabel("Epoch Number")
	plt.ylabel("Epoch Loss")
	plt.title("Epoch Loss Curve")

	# keeping track of the learning rate
	graph = tf.get_default_graph()
	lr_test = graph.get_tensor_by_name("LR_Decaying:0")

	# keeping track of training time
	start = time.time()

	# enabling dynamic allocation of GPU memory
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		strikes = 0		# metric for the early stopping. each time the delta loss falls under desired threshold, add a strike
		E_val_loss_prev = 0

		# training model
		# Prefix Guide: E refers to epoch scope, while B refers to batch scope
		for epoch in range(n_epochs):
			E_train_loss = 0
			for batch in range(int(len(train_x)/batch_size)):	# feeding in training batches
				B_train_x = train_x[batch*batch_size:min((batch+1)*batch_size, len(train_x)-1)]
				B_train_y = train_y[batch*batch_size:min((batch+1)*batch_size, len(train_y)-1)]

				opt, B_train_loss = sess.run([optimizer, cost], feed_dict={x: B_train_x, y: B_train_y})
				E_train_loss += B_train_loss

			E_val_loss = 0
			for batch in range(int(len(test_x)/batch_size)):	# feeding in testing batches
				B_test_x = test_x[batch*batch_size:min((batch+1)*batch_size, len(test_x)-1)]
				B_test_y = test_y[batch*batch_size:min((batch+1)*batch_size, len(test_y)-1)]

				B_test_loss = sess.run(cost, feed_dict={x: B_test_x, y: B_test_y})
				E_val_loss += B_test_loss

			# stats per epoch
			print("Epoch {}/{}." .format(epoch+1, n_epochs))
			print("Epoch Loss     : {}" .format(E_train_loss))
			print("Validation Loss: {}" .format(E_val_loss))
			print("Learning Rate: {}" .format(lr_test.eval()))

			test_pred = sess.run(prediction, feed_dict={x: test_x[:5], y: test_y[:5]})
			print("Predictions[0]: {}, {}" .format(test_pred[0], test_y[0]))
			print("Predictions[1]: {}, {}" .format(test_pred[1], test_y[1]))
			print("Predictions[2]: {}, {}" .format(test_pred[2], test_y[2]))
			print("Predictions[3]: {}, {}" .format(test_pred[3], test_y[3]))

			plt.scatter(epoch, E_train_loss)
			plt.pause(0.05)

			if (epoch >= 1):
				strikes = early_stop([E_val_loss_prev, E_val_loss], strikes, 0.01, "validation")
				E_val_loss_prev = E_val_loss
				print("Strikes: {}" .format(strikes))
				if (strikes == 5):
					print("Early Stop at Epoch:{}/{}" .format(epoch, n_epochs))
					break
			print("\n")

		end = time.time()
		print("\nTraining Done! Time Elapsed: {} minutes" .format((end - start)/60.0))
		plt.show()

		# Saving model
		print("Saving Model: \"PNN_V2_MODEL_{}\"" .format(PNN_VERSION))
		saver.save(sess, "./PNN_V2_Model_{}".format(PNN_VERSION))
		plt.close("all")


if __name__ == '__main__':
	ConvNN_Train(x)

