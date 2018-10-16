import numpy as np
import tensorflow as tf
import cv2
import time
from PIL import Image
from ConvNet_Model.PilotNetModel_v2 import cnn_model


tf.reset_default_graph()

WIDTH = 80
HEIGHT = 60
n_outputs = 1
pool_s = 2

x = tf.placeholder("float", [None, HEIGHT, WIDTH])
model = cnn_model(x, WIDTH, HEIGHT, n_outputs, is_training=False)

loader = tf.train.Saver()

print("Loading Data")
data = "test"
if (data == "validation"):
	test_data = np.load("./Data/validation_set.npy")
elif (data == "test"):
	test_data = np.load("./Data/test_data.npy")
display_data = np.load("./Data/display_data.npy")

# model information
version = "v1"
model_name = "./Model_Data/PNN_{}" .format(version)

def calc_error(prediction, label):
	error = (prediction - label)**2
	return error


def display_prediction(dst_img, src_img, prediction):
	# pasting steering wheel image onto road image
	road_x, road_y = dst_img.size
	wheel_x, wheel_y = src_img.size
	dst_img.paste(src_img, (int(road_x / 2 - wheel_x / 2), int(road_y / 2 + 1.5 * wheel_y)))
	road_img = np.array(dst_img)

	# pasting the steering angle onto road image
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(road_img, "%.2f Degrees" % (prediction[0]*180), (50, 50), font, 1, (0,0,255), 2)
	return road_img

def process_steering(pred_angle, steering_wheel_img):
	pred_angle = pred_angle[0] * 180
	rot_mat = cv2.getRotationMatrix2D((40, 40), -1 * pred_angle, 1)
	pred_steering_wheel = cv2.warpAffine(steering_wheel_img, rot_mat, (81, 81))
	return pred_steering_wheel

def main():
	last_time = time.time()

	# loading steering wheel and preparing for roatation
	steering_wheel_img = np.load("steering_wheel.npy")

	training_variables = tf.trainable_variables()	# getting list of trainable variables defined in the model

	print("----- Loading Weights -----")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	# allow dynamic allocation of memory
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		# load up saved model
		graph = tf.get_default_graph()
		loader.restore(sess, model_name)

		i = 0
		running_error = 0
		for screen in test_data:
			# making prediction
			prediction = sess.run(model, feed_dict={x: screen[0].reshape(-1, HEIGHT, WIDTH)})[0]

			steering_wheel = process_steering(prediction, steering_wheel_img)

			# adding prediction onto test photos
			road_img = Image.fromarray(display_data[i], "RGB")	# converting numpy array to Image format
			wheel_img = Image.fromarray(steering_wheel, "RGB")
			test_img = display_prediction(road_img, wheel_img, prediction)
			cv2.imshow("Model_Test", test_img)

			# calculating error
			running_error += calc_error(prediction, screen[1])

			i += 1
			if (i == len(display_data) - 1):
				cv2.destroyAllWindows()
				break
			if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break

		print("Average difference: {}" .format(running_error/i))


if __name__ == "__main__":
	main()



