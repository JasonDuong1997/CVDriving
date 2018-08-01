import tensorflow as tf
from ConvNet_Model.Model import ConvNN_Model
import tflearn
import numpy as np

training_data = np.load("training_data_balanced.npy")

learning_rate = 1e-3
batch_size = 128  	# number of images per cycle (in the power of 2 because # of physical processors is similar)
n_epochs = 8	 	# number of epochs
n_outputs = 3	  	# number of outputs
pool_s = 2			# maxpool stride

WIDTH = 80
HEIGHT = 62

x = tf.placeholder("float", [None, HEIGHT, WIDTH])
y = tf.placeholder("float", [None, n_outputs])

def ConvNN_Train(x):
	prediction = ConvNN_Model(x, WIDTH, HEIGHT, n_outputs, pool_s)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))

	# separating out the data into training and validation set
	x_set = [i[0] for i in training_data]
	train_x = x_set[:-500]
	test_x = x_set[-500:]

	y_set = [i[1] for i in training_data]
	train_y = y_set[:-500]
	test_y = y_set[-500:]

	print("train/test X: {}, {}" .format(len(train_x), len(test_x)))
	print("train/test Y: {}, {}" .format(len(train_y), len(test_y)))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		# training
		for epoch in range(n_epochs):
			epoch_loss = 0
			for batch in range(int(len(train_x)/batch_size)):
				batch_x = train_x[batch*batch_size:min((batch+1)*batch_size, len(train_x)-1)]
				batch_y = train_y[batch*batch_size:min((batch+1)*batch_size, len(train_y)-1)]

				opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

				epoch_loss += loss

			epoch_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
			print("Epoch {}/{}." .format(epoch+1, n_epochs))
			print("Loss: {}, Accuracy: {}" .format(epoch_loss, epoch_acc))

		print("\nTraining Done!")
		print("Final Accuracy: {}" .format(accuracy.eval({x: test_x, y: test_y})))

		# Saving model
		print("Saving Model")
		saver.save(sess, "./CNN_Model")

if __name__ == '__main__':
	ConvNN_Train(x)
