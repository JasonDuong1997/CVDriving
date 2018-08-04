import numpy as np
import tensorflow as tf
import cv2
from ConvNet_Model.Model import ConvNN_Model
import time
from PIL import ImageGrab
from GameControls import PressKey, ReleaseKey, W, A, D
from KeyLogger import KeyCheck
import win32gui

WIDTH = 80
HEIGHT = 62
n_outputs = 3
pool_s = 2

hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[1]
win_w = rect[2] - win_x
win_h = rect[3] - win_y


WIDTH = int(win_w/10)
HEIGHT = int(win_h/10)

x = tf.placeholder("float", [None, HEIGHT, WIDTH])
model = tf.nn.softmax(ConvNN_Model(x, WIDTH, HEIGHT, n_outputs, pool_s))

def straight():
	ReleaseKey(A)
	ReleaseKey(D)
	PressKey(W)

def left():
	ReleaseKey(D)
	# ReleaseKey(W)
	PressKey(A)

def right():
	ReleaseKey(A)
	# ReleaseKey(W)
	PressKey(D)


def main():
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()
	frame_loop = 0

	y_van = 11*win_h/50

	print("----- Loading Weights -----")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	# allow dynamic allocation of memory
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph("./CNN_Model.meta")
		saver.restore(sess, tf.train.latest_checkpoint("./"))

		for i in range(0,3):
			print("On the count of 3: {}" .format(i))
			time.sleep(.5)

		paused = False

		while(True):
				if (not paused):
					# 800x600 windowed mode
					# bbox(x, y, width, height)
					vertices = np.array([[0, win_h], [0, 7*win_h/16], [win_w/4, y_van], [3*win_w/4, y_van], [win_w, 7*win_h/16], [win_w, win_h], [5*win_w/8, 11*win_h/16], [3*win_w/8, 11*win_h/16]], np.int32)
					image = np.array(ImageGrab.grab(bbox=(win_x, win_y, rect[2], rect[3])))  # grabbing screen into a numpy array	// for GTAV
					grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					screen = cv2.resize(grey, (WIDTH,HEIGHT))


					print("loop took {} seconds " .format(time.time()-last_time))
					last_time = time.time()

					prediction = model.eval({x: screen.reshape(-1, HEIGHT, WIDTH)})[0]
					move = list(np.around(prediction))
					print("Move: {}" .format(move))

					# starting the gas
					straight()
					if move == [1,0,0]:
						left()
						print("Steering: Left")
					elif move == [0,1,0]:
						straight()
						print("Steering: Straight")
					elif move == [0,0,1]:
						right()
						print("Steering: Right")


				keys = KeyCheck()
				if 'T' in keys:
					if paused:
						paused = False
						time.sleep(.5)
					else:
						pause = True
						ReleaseKey(A)
						ReleaseKey(W)
						ReleaseKey(D)
						time.sleep(.5)
						break

				# for counting the number of frames
				frames = frames + 1
				if (frames == 1000):
					break

				print("Number of frames: {} " .format(frames))
				if cv2.waitKey(25) & 0xFF == ord('q'):
						cv2.destroyAllWindows()
						final_time = time.time()
						print("Average FPS: {} " .format(frames/(final_time - elapsed)))
						break

if __name__ == "__main__":
	main()



