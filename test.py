import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui as pag
import win32gui

hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[0]
win_w = rect[2] - win_x
win_h = rect[3] - win_y
y_van = int(11*win_h/50)

while (1):
	image = np.array(ImageGrab.grab(bbox=(win_x, win_y, win_w, win_h)))  # grabbing screen into a numpy array	// for GTAV

	shadow_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	shadow_blur = cv2.GaussianBlur(shadow_grey, (5,5), 0)         									 # apply blur to smooth
	shadow_edges = cv2.Canny(shadow_blur, threshold1=100, threshold2=170)

	hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	l = hsl[:,:,1]
	l_threshed = cv2.inRange(l, 70, 100)

	test = shadow_edges & l_threshed

	cv2.imshow("L", l_threshed)
	cv2.imshow("shadow", test)

	# LINES
	"""lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, np.array([]), 30, 60)
	for line in lines:
		coords = line[0]
		cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), (255,255,255), thickness=2)
	"""




	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break


cv2.destroyAllWindows()


