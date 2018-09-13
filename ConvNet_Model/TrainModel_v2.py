import tensorflow as tf
from ConvNet_Model.PilotNetModel_v2 import  PilotNetV2_Model
import matplotlib.pyplot as plt
import numpy as np
import time

training_data = np.load("udacity_trainingData_processed.npy")

learning_rate = 3e-5
test_size = int(len(training_data)*0.01)
batch_size = 128  	# number of images per cycle (in the power of 2 because # of physical processors is similar)
n_epochs = 450	 	# number of epochs
n_outputs = 1	  	# number of outputs
pool_s = 2			# maxpool stride

WIDTH = 80
HEIGHT = 60

x = tf.placeholder("float", [None, HEIGHT, WIDTH, 3])
y = tf.placeholder("float", [None, n_outputs])

PNN_VERSION = "1.0"


def round_decimal(num, n_positions):
	format = "%." + str(n_positions) + "f"
	return float(format % num)


def ConvNN_Train(x):
	# OPERATIONS
	prediction = PilotNetV2_Model(x, WIDTH, HEIGHT, n_outputs, pool_s)
	training_variables = tf.trainable_variables()	# getting list of trainable variables defined in the model
	cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y))) + tf.add_n([tf.nn.l2_loss(variable) for variable in training_variables if "B" not in variable.name])*learning_rate

	# optimizer with normalization
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08).minimize(cost)


	# separating out the data into training and validation set
	x_set = [i[0] for i in training_data]
	train_x = x_set[:-test_size]
	test_x = x_set[-test_size:]
	y_set = [i[1] for i in training_data]
	train_y = y_set[:-test_size]
	test_y = y_set[-test_size:]
	print("train/test X: {}, {}" .format(len(train_x), len(test_x)))
	print("train/test Y: {}, {}" .format(len(train_y), len(test_y)))

	# plotting the loss
	plt.figure(figsize=(15,8))
	plt.axis([0, n_epochs, 0, 50])
	plt.grid(True)
	plt.xticks(np.arange(0, n_epochs, 10))
	plt.yticks(np.arange(0, 50, 2))
	plt.xlabel("Epoch Number")
	plt.ylabel("Epoch Loss")
	plt.title("Epoch Loss Curve")

	# enabling dynamic allocation of GPU memory
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		#sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		# training
		for epoch in range(n_epochs):
			epoch_loss = 0
			for batch in range(int(len(train_x)/batch_size)):
				batch_x = train_x[batch*batch_size:min((batch+1)*batch_size, len(train_x)-1)]
				batch_y = train_y[batch*batch_size:min((batch+1)*batch_size, len(train_y)-1)]

				opt, loss = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += loss

			val_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
			test_pred = sess.run(prediction, feed_dict={x: test_x, y: test_y})
			print("Epoch {}/{}." .format(epoch+1, n_epochs))
			print("Epoch Loss     : {}" .format(epoch_loss))
			print("Validation Loss: {}" .format(val_loss))
			print("Predictions[0]: {}, {}" .format(test_pred[0], test_y[0]))
			print("Predictions[1]: {}, {}" .format(test_pred[50], test_y[50]))
			print("Predictions[2]: {}, {}" .format(test_pred[100], test_y[100]))
			# acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
			# print("Epoch Acc: {}" .format(acc))

			plt.scatter(epoch, epoch_loss)
			plt.pause(0.05)
		plt.show()
		print("\nTraining Done!")

		# Saving model
		print("Saving Model: \"PNN_V2_MODEL_{}\"" .format(PNN_VERSION))
		saver.save(sess, "./PNN_V2_Model_{}".format(PNN_VERSION))
		plt.close("all")


if __name__ == '__main__':
	start = time.time()
	ConvNN_Train(x)
	end = time.time()
	print("Total time: {}" .format(end - start))