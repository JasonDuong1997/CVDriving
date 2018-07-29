import numpy as np
from PIL import ImageGrab
import cv2
import time
from GameControls import PressKey, ReleaseKey, W, A, D
from KeyLogger import KeyCheck
import win32gui
from AlexNet_Model.TrainData import alexnet
import tensorflow as tf


hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[1]
win_w = rect[2] - win_x
win_h = rect[3] - win_y


WIDTH = int(win_w/10)
HEIGHT = int(win_h/10)


LR = 1e-3		# learning rate
EPOCHS = 8
MODEL_NAME = "pygta5-SmartCar-{}-{}-{}-epochs.model" .format(LR, "alexnet", EPOCHS)

def straight():
	ReleaseKey(A)
	ReleaseKey(D)
	PressKey(W)

def left():
	ReleaseKey(W)
	ReleaseKey(D)
	PressKey(A)

def right():
	ReleaseKey(A)
	ReleaseKey(W)
	PressKey(D)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()
	frame_loop = 0

	y_van = 11*win_h/50

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

				prediction = model.predict(screen.reshape(1, WIDTH, HEIGHT, 1))[0]
				moves = list(np.around(prediction))
				print(moves, prediction)


				if moves == [1,0,0]:
					left()
				elif moves == [0,1,0]:
					straight()
				elif moves == [0,0,1]:
					right()


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