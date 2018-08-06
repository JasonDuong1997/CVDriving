import numpy as np
from PIL import ImageGrab
import cv2
import time
from KeyLogger import KeyCheck
import win32gui
import os


hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[1]
win_w = rect[2] - win_x
win_h = rect[3] - win_y

image_type = "grey"
data_version = "yuv"

# convert to 1-hot array
def keys_to_output(keys):
	# [A,W,D]
	output = [0, 0, 0]

	if 'A' in keys:
		output[0] = 1
	elif 'D' in keys:
		output[2] = 1
#	elif 'W' in keys:
#		output[1] = 1

	return output


file_name = "training_data.npy"

if os.path.isfile(file_name):
	print("File exists, loading previous data")
	training_data = list(np.load(file_name))
	print(len(training_data))
else:
	print("File does not exist. Starting fresh")
	training_data = []

def main():
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()
	frame_loop = 0

	save_count = 0

	y_van = 11*win_h/50

	for i in range(0,3):
		print("On the count of 3: {}" .format(i))
		time.sleep(.5)

	while(True):
			# 800x600 windowed mode
			# bbox(x, y, width, height)
			vertices = np.array([[0, win_h], [0, 7*win_h/16], [win_w/4, y_van], [3*win_w/4, y_van], [win_w, 7*win_h/16], [win_w, win_h], [5*win_w/8, 11*win_h/16], [3*win_w/8, 11*win_h/16]], np.int32)

			image = np.array(ImageGrab.grab(bbox=(win_x, win_y, rect[2], rect[3])))  # grabbing screen into a numpy array	// for GTAV

			if (image_type == "grey"):
				grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				screen = cv2.resize(grey, (int(win_w/10),int(win_h/10)))
			elif (image_type == "yuv"):
				yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
				screen = cv2.resize(yuv, (int(win_w/10),int(win_h/10)))

			keys = KeyCheck()
			output = keys_to_output(keys)

			if (output != [0, 0, 0]):
				training_data.append([screen, output])
				save_count += 1

			#print("loop took {} seconds " .format(time.time()-last_time))
			last_time = time.time()

			if (save_count == 500):
				print(len(training_data))
				# np.save(file_name, training_data)
				break

			# for counting the number of frames
			frames = frames + 1
			if (frames == 2000):
				print(len(training_data))
				print("Frames Saved: {}" .format(save_count))
				np.save(file_name, training_data)
				break

			print("Number of frames: {} " .format(frames))
			if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					final_time = time.time()
					print("Average FPS: {} " .format(frames/(final_time - elapsed)))
					break

if __name__ == "__main__":
	main()
