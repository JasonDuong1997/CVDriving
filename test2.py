import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui as pag
import GameControls as GInput
import win32gui
import math
import traceback
import operator

hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[0]
win_w = rect[2] - win_x
win_h = rect[3] - win_y
y_van = int(11*win_h/50)
def Order_PTPoints(points):
	# four coordinates of a trapezoid
	trapezoid = np.zeros((4,2), dtype = "float32")

	sorted_points = sorted(points, key=operator.itemgetter(1))

	if (sorted_points[0][0] < sorted_points[1][0]):
		trapezoid[0] = sorted_points[0]
		trapezoid[1] = sorted_points[1]
	else:
		trapezoid[0] = sorted_points[1]
		trapezoid[1] = sorted_points[0]

	if (sorted_points[2][0] < sorted_points[3][0]):
		trapezoid[2] = sorted_points[3]
		trapezoid[3] = sorted_points[2]
	else:
		trapezoid[2] = sorted_points[2]
		trapezoid[3] = sorted_points[3]

	return trapezoid


def Birds_Eye_View(image):
	points = ([13.65*win_w/32, 14.5*win_h/32],[18.35*win_w/32, 14.5*win_h/32],[win_w, 19*win_h/32], [0, 19*win_h/32])
	trapezoid = Order_PTPoints(points)
	(topL, topR, botL, botR) = trapezoid

	# calculate width of transformed image
	w1 = np.sqrt(math.pow((botR[0] - botL[0]),2) + math.pow((botR[1] - botL[1]),2))
	w2 = np.sqrt(math.pow((topR[0] - topL[0]),2) + math.pow((topR[1] - topL[1]),2))
	width = max(int(w1), int(w2))

	# calculate height of the transformaed image
	h1 = np.sqrt(math.pow((topR[0] - botR[0]),2) + math.pow((topR[1] - botR[1]),2))
	h2 = np.sqrt(math.pow((topL[0] - botL[0]),2) + math.pow((topL[1] - botL[1]),2))
	height = max(int(h1), int(h2))

	# bounds of the destination image
	dst = np.array(([0,0], [width-1,0], [width-1, height-1], [0,height]), dtype = "float32")

	# transform the image to birds eye view
	M = cv2.getPerspectiveTransform(trapezoid, dst=dst)
	warped = cv2.warpPerspective(image, M, (width,height))
	return warped



while (True):
	image = np.array(ImageGrab.grab(bbox=(win_x, win_y, rect[2], rect[3])))  # grabbing screen into a numpy array	// for GTAV
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	processed_image = Birds_Eye_View(image)

	cv2.imshow("BIRDS EYE VIEW", processed_image)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break