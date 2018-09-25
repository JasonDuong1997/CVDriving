import tensorflow as tf
from ConvNet_Model.PilotNetModel_v2 import  PilotNetV2_Model
import matplotlib.pyplot as plt
import numpy as np
import time

### HELPER FUNCTIONS ###
def k_fold_splitter(training_data, k=5):	# splitting training data into k equal parts
	print("K-Fold Split with K={}" .format(k))
	fold_len = int(len(training_data)/k)

	split_data = []
	for i in range(k):
		start_index = i*fold_len
		split_data.append(training_data[start_index:(start_index + fold_len - 1)])

	return split_data

def k_fold_selector(split_data, index):	# sectioning out train and test set
	print("K-Fold Selector with Index={}" .format(index))
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

	return train_x, train_y, test_x, test_y

def early_stop(loss_check, strikes, strikeout, threshold, loss_monitor="validation"):	# checks the condition on when to finish training
	if (loss_monitor == "validation"):
		delta_loss = loss_check[1] - loss_check[0]
		# if the change in loss is less than [tolerance]% or if the loss increased, add a strike
		if (abs(delta_loss) < abs(threshold*loss_check[0]) or delta_loss > 0):
			strikes += 1
			if (strikes == strikeout):
				strikes = -1
		else:
			strikes = 0
	return strikes

def cyclical_lr(epoch, amplitude, period):	#todo make this work with decay LR
	return np.sin(epoch*2*np.pi/period)*amplitude


### GLOBAL VARIABLES###
print("Loading Data...")
loaded_data = np.load("./Data/udacity_TD_proc_aug.npy")

# separating out the data into training and validation set
print("Separating Data into Training and Test Sets...")
k_splits = 10
k_split_data = k_fold_splitter(loaded_data, k_splits)
train_x, train_y, test_x, test_y = k_fold_selector(k_split_data, 1)
print("[X] Train Size: {}. Test Size: {}".format(len(train_x), len(test_x)))
print("[Y] Train Size: {}. Test Size: {}".format(len(train_y), len(test_y)))

# training variables
batch_size = 64  	# number of images per cycle (in the power of 2 because # of physical processors is similar)
n_epochs = 3000 	# number of epochs
n_outputs = 1	  	# number of outputs

# deciding on learning rate using Grid Search
# *********************************** #
# Learning Rate List with 300 Epochs  #
# *********************************** #
# - using regular dataset
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
initial_learning_rate = 5e-5
epsilon = 5e-6
decay_rate = 0.80
steps_per_epoch = len(train_x)/batch_size
epochs_per_decay = 200
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, epochs_per_decay*steps_per_epoch, decay_rate, staircase=True, name="LR_Decaying")

# image input dimensions
WIDTH = 80
HEIGHT = 60
# input placeholder tensors
x = tf.placeholder("float", [None, HEIGHT, WIDTH, 3])
y = tf.placeholder("float", [None, n_outputs])

# model information
version = "v2"
model_name = "./Model_Data/PNN_{}" .format(version)

### TRAINING FUNCTION ###
# Specifications
# Model: 				PilotNetV2
# Cost Function: 		Mean Squared Loss with L2 Loss Weight Penalization
# Optimizer: 			AdamOptimizer
# Learning Rate Mod: 	Exponential Decay
# Special Functions: 	Early Stopping, K-Fold Cross-Validation
def ConvNN_Train(x):
	# declaring operations
	prediction = PilotNetV2_Model(x, WIDTH, HEIGHT, n_outputs, is_training=True)
	training_variables = tf.trainable_variables()	# getting list of trainable variables defined in the model
	cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y))) + tf.add_n([tf.nn.l2_loss(variable) for variable in training_variables if "B" not in variable.name])*learning_rate		# penalizing large weights with L2 loss

	# creating optimizer with normalization
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(cost, global_step=global_step)

	# creating plot to graph the loss
	x_max = n_epochs
	x_scale = n_epochs/20
	y_max = 0.5
	y_scale = 0.02
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
		is_saved = False

		# training model
		# Prefix Guide: "E" refers to epoch scope, while "B" refers to batch scope
		print("Training Starting!")
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
			print("\nEpoch {}/{}." .format(epoch+1, n_epochs))
			print("Epoch Loss     : {}" .format(E_train_loss))
			print("Validation Loss: {}" .format(E_val_loss))
			print("Learning Rate: {}" .format(lr_test.eval()))

			test_pred = sess.run(prediction, feed_dict={x: test_x[:5], y: test_y[:5]})
			print("Predictions[0]: {}, {}" .format(test_pred[0], test_y[0]))
			print("Predictions[1]: {}, {}" .format(test_pred[1], test_y[1]))
			print("Predictions[2]: {}, {}" .format(test_pred[2], test_y[2]))
			print("Predictions[3]: {}, {}" .format(test_pred[3], test_y[3]))

			plt.scatter(epoch, E_val_loss)
			plt.pause(0.05)

			if (epoch >= 1):
				strikes = early_stop([E_val_loss_prev, E_val_loss], strikes, 5, 0.01, "validation")
				E_val_loss_prev = E_val_loss
				print("Strikes: {}" .format(strikes))
				if (strikes == 1):	# saving model right when validation loss starts to increase
					print("Saving Model Checkpoint")
					saver.save(sess, model_name)
				if (strikes == -1):	# strikeout condition
					print("Early Stop at Epoch:{}/{}" .format(epoch, n_epochs))
					is_saved = True
					end = time.time()
					print("\nTraining Done! Time Elapsed: {} minutes".format((end - start) / 60.0))
					plt.show()
					break

		end = time.time()
		print("\nTraining Done! Time Elapsed: {} minutes" .format((end - start)/60.0))
		plt.show()

		# Saving model
		if (not is_saved):
			print("Saving Model: {}" .format(model_name))
			saver.save(sess, model_name)
		else:
			print("Model Saved: {}" .format(model_name))

		plt.close("all")
# Early Stopping Results
# -using augmented dataset
# 		Epoch		Train_Loss		Val_Loss		ILR			FLR
# 1.	298			1.356			0.538			8.0e-5		6.4e-5
# 2.	120			3.851			0.778			5.0e-5		5.0e-5
if __name__ == '__main__':
	ConvNN_Train(x)

