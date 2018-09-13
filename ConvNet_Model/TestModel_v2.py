import numpy as np
import tensorflow as tf
import cv2
import time
from ConvNet_Model.PilotNetModel_v2 import PilotNetV2_Model

WIDTH = 80
HEIGHT = 60
n_outputs = 1
pool_s = 2

x = tf.placeholder("float", [None, HEIGHT, WIDTH])
model = PilotNetV2_Model(x, WIDTH, HEIGHT, n_outputs, pool_s, is_training=False)

PNN_VERSION = "1.0"

test_data = np.load("test_data.npy")
display_data = np.load("display_data.npy")


def calc_error(prediction, label):
	error = abs(prediction - label)
	return error


def main():
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()
	frame_loop = 0

	print("----- Loading Weights -----")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	# allow dynamic allocation of memory
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		saver = tf.train.import_meta_graph("./PNN_V2_Model_{}.meta".format(PNN_VERSION))
		saver.restore(sess, tf.train.latest_checkpoint("./"))

		for i in range(0,3):
			print("On the count of 3: {}" .format(i))
			time.sleep(.5)


		print("loop took {} seconds " .format(time.time()-last_time))
		last_time = time.time()

		i = 0
		running_error = 0
		for screen in test_data:
			# making prediction
			prediction = model.eval({x: screen[0].reshape(-1, HEIGHT, WIDTH)})[0]

			# displaying prediction on test feed
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(display_data[i], str(prediction) + str(screen[1]), (50, 50), font, 1, (0,0,255), 2)
			print(str(prediction) + " : " + str(screen[1]))
			running_error += calc_error(prediction, screen[1])
			cv2.imshow("image", display_data[i])
			i += 1
			if (i == 100):
				break
			if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
		print("Average difference: {}" .format(running_error/i))


if __name__ == "__main__":
	main()



