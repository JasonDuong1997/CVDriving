import numpy as np
from PIL import ImageGrab
import cv2
import time
import GameControls as GInput
from KeyLogger import KeyCheck
import win32gui
import os


hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[1]
win_w = rect[2] - win_x
win_h = rect[3] - win_y

############## Driving functions  ####################
def applyGas(apply=True):
	if (apply == True):
		GInput.PressKey(GInput.Key.W)
	else:
		GInput.ReleaseKey(GInput.Key.W)


def applyBreaks(apply=True):
	if (apply == True):
		GInput.PressKey(GInput.Key.S)
	else:
		GInput.ReleaseKey(GInput.Key.S)


def steer_left():
	GInput.ReleaseKey(GInput.Key.D)
	GInput.PressKey(GInput.Key.A)


def steer_right():
	GInput.ReleaseKey(GInput.Key.A)
	GInput.PressKey(GInput.Key.D)

def release_all_controls():
	#GInput.ReleaseKey(GInput.Key.W)
	GInput.ReleaseKey(GInput.Key.A)
	GInput.ReleaseKey(GInput.Key.S)
	GInput.ReleaseKey(GInput.Key.D)
########################################################

# convert to 1-hot array
def keys_to_output(keys):
	# [A,W,D]
	output = [0, 0, 0]

	if 'A' in keys:
		output[0] = 1
	elif 'D' in keys:
		output[2] = 1
	elif 'W' in keys:
		output[1] = 1

	return output


file_name = "training_data.npy"

if os.path.isfile(file_name):
	print("File exists, loading previous data")
	training_data = list(np.load(file_name))
else:
	print("File does not exist. Starting fresh")
	training_data = []

def main():
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()
	frame_loop = 0

	y_van = 11*win_h/50

	for i in range(0,3):
		print("On the count of 3: {}" .format(i))
		time.sleep(.5)

	while(True):
			# 800x600 windowed mode
			# bbox(x, y, width, height)
			vertices = np.array([[0, win_h], [0, 7*win_h/16], [win_w/4, y_van], [3*win_w/4, y_van], [win_w, 7*win_h/16], [win_w, win_h], [5*win_w/8, 11*win_h/16], [3*win_w/8, 11*win_h/16]], np.int32)

			image = np.array(ImageGrab.grab(bbox=(win_x, win_y, rect[2], rect[3])))  # grabbing screen into a numpy array	// for GTAV
			grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			screen = cv2.resize(grey, (int(win_w/10),int(win_h/10)))

			keys = KeyCheck()
			output = keys_to_output(keys)
			training_data.append([screen, output])

			print("loop took {} seconds " .format(time.time()-last_time))
			last_time = time.time()

			if (len(training_data) % 500 == 0):
				print(len(training_data))
				np.save(file_name, training_data)



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
