import numpy as np
import tensorflow as tf
import cv2
import time
from ConvNet_Model.PilotNetModel_v2 import PilotNetV2_Model


tf.reset_default_graph()

WIDTH = 80
HEIGHT = 60
n_outputs = 1
pool_s = 2

x = tf.placeholder("float", [None, HEIGHT, WIDTH])
model = PilotNetV2_Model(x, WIDTH, HEIGHT, n_outputs, is_training=False)

loader = tf.train.Saver()

PNN_VERSION = "1.0"

print("Loading Data")
data = "test"
if (data == "validation"):
	test_data = np.load("validation_set.npy")
elif (data == "test"):
	test_data = np.load("test_data.npy")
display_data = np.load("display_data.npy")



def calc_error(prediction, label):
	error = (prediction - label)**2
	return error


def main():
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()
	frame_loop = 0

	training_variables = tf.trainable_variables()	# getting list of trainable variables defined in the model

	print("----- Loading Weights -----")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	# allow dynamic allocation of memory
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		# load up saved model
		graph = tf.get_default_graph()
		print(graph.get_tensor_by_name("B_fc4:0").eval())
		loader.restore(sess, "./PNN_V2_Model_{}".format(PNN_VERSION))

		print("loop took {} seconds " .format(time.time()-last_time))
		last_time = time.time()

		i = 0
		running_error = 0
		for screen in test_data:
			# making prediction
			# prediction = model.eval({x: screen[0].reshape(-1, HEIGHT, WIDTH)})[0]
			prediction = sess.run(model, feed_dict={x: screen[0].reshape(-1, HEIGHT, WIDTH)})[0]

			# adding prediction onto test photos
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(display_data[i], str(prediction*-180) + str(screen[1]*-180), (50, 50), font, 1, (0,0,255), 2)
			cv2.imshow("image", display_data[i])
			print(str(prediction) + " : " + str(screen[1]))

			# calculating error
			running_error += calc_error(prediction, screen[1])

			i += 1
			if (i == 300):
				cv2.destroyAllWindows()
				break
			if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
		print("Average difference: {}" .format(running_error/i))


if __name__ == "__main__":
	main()



