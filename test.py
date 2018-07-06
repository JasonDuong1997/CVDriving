import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui as pag
import win32gui

hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[1]
win_w = rect[2] - win_x
win_h = rect[3] - win_y
y_van = int(11 * win_h / 50)
print(win_w, win_h)

while (1):

	image = np.array(ImageGrab.grab(bbox=(win_x, win_y, rect[2], rect[3])))  # grabbing screen into a numpy array	// for GTAV

	r = image[:,:,0]
	r_threshed = cv2.inRange(r, 210,255)


	hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	s = hsl[:,:,2]
	s_threshed = cv2.inRange(s, 45, 100)


	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l = lab[:,:,0]
	b = lab[:,:,2]
	l_threshed = cv2.inRange(l, 150, 225)
	b_threshed = cv2.inRange(b, 110, 120)
	lb_threshed = l_threshed & b_threshed

	luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
	u = luv[:,:,1]
	u_threshed = cv2.inRange(u, 85, 180)
	test = s_threshed | b_threshed | r_threshed
	cv2.imshow("TEST", lb_threshed)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break


cv2.destroyAllWindows()


