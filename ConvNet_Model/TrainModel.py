import tensorflow as tf
from ConvNet_Model.Model import ConvNN_Model
from ConvNet_Model.PilotNetModel import  PilotNet_Model
import tflearn
import numpy as np

data_version = "yuv"
training_data = np.load("training_data_shuffled.npy")
testing_data = np.load("test_data_yuv.npy")

learning_rate = 5e-5
test_size = int(len(training_data)*0.03)
batch_size = 128  	# number of images per cycle (in the power of 2 because # of physical processors is similar)
n_epochs = 60	 	# number of epochs
n_outputs = 3	  	# number of outputs
pool_s = 2			# maxpool stride

WIDTH = 80
HEIGHT = 62

if (data_version == "yuv"):
	x = tf.placeholder("float", [None, HEIGHT, WIDTH, 3])
else:
	x = tf.placeholder("float", [None, HEIGHT, WIDTH])
y = tf.placeholder("float", [None, n_outputs])

CNN_VERSION = "1.1"
PNN_VERSION = "1.0"

def ConvNN_Train(x):
	if (data_version == "yuv"):
		prediction = PilotNet_Model(x, WIDTH, HEIGHT, n_outputs, pool_s)
	else:
		prediction = ConvNN_Model(x, WIDTH, HEIGHT, n_outputs, pool_s)


	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))

	# separating out the data into training and validation set
	if (data_version == "yuv"):
		train_x = [i[0] for i in training_data]
		train_y = [i[1] for i in training_data]
		test_x = [i[0] for i in testing_data]
		test_y = [i[1] for i in testing_data]
	else:
		x_set = [i[0] for i in training_data]
		train_x = x_set[:-test_size]
		test_x = x_set[-test_size:]

		y_set = [i[1] for i in training_data]
		train_y = y_set[:-test_size]
		test_y = y_set[-test_size:]

	print("train/test X: {}, {}" .format(len(train_x), len(test_x)))
	print("train/test Y: {}, {}" .format(len(train_y), len(test_y)))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		running_acc = 0

		# training
		for epoch in range(n_epochs):
			epoch_loss = 0
			for batch in range(int(len(train_x)/batch_size)):
				batch_x = train_x[batch*batch_size:min((batch+1)*batch_size, len(train_x)-1)]

				batch_y = train_y[batch*batch_size:min((batch+1)*batch_size, len(train_y)-1)]

				"""
				batch = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
				"""

				opt, loss = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				running_acc += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

				epoch_loss += loss

			epoch_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
			print("Epoch {}/{}." .format(epoch+1, n_epochs))
			print("Epoch Loss: {}, Epoch Accuracy: {}" .format(epoch_loss/test_size, epoch_acc))
			print("Running Accuracy: {}" .format(running_acc/int(len(train_x)/batch_size)))
		print("\nTraining Done!")
		print("Final Accuracy: {}" .format(accuracy.eval({x: test_x, y: test_y})))

		# Saving model
		if (data_version == "yuv"):
			print("Saving Model: \"PNN_MODEL{}\"" .format(PNN_VERSION))
			saver.save(sess, "./PNN_Model{}".format(PNN_VERSION))
		else:
			print("Saving Model: \"CNN_MODEL{}\"" .format(CNN_VERSION))
			saver.save(sess, "./CNN_Model{}" .format(CNN_VERSION))

if __name__ == '__main__':
	ConvNN_Train(x)
