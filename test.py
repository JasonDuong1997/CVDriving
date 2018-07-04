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

	red = image[:,:,0]
	green = image[:,:,1]
	red_threshed = cv2.inRange(red, 240, 255)
	green_threshed = cv2.inRange(green, 230, 255)
	red_green_mask = cv2.bitwise_and(src1=red_threshed, src2=green_threshed)
	mask1 = red_green_mask>0
	red_green_f = np.zeros_like(image, np.uint8)
	red_green_f[mask1] = image[mask1]
	rg_f_gray = cv2.cvtColor(red_green_f, cv2.COLOR_BGR2GRAY)

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	v = image[:,:,2]
	v_threshed = cv2.inRange(v, 170,255)
	cv2.imshow("V", v_threshed)


	hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	s = hsl[:,:,2]
	s_thresh = cv2.inRange(s, 200, 255)
	s_mask = cv2.bitwise_and(hsl, hsl, mask=s_thresh)
	mask2 = s_mask>0
	s_mask_f = np.zeros_like(image, np.uint8)
	s_mask_f[mask2] = image[mask2]

	test = cv2.bitwise_or(src1=red_green_mask, src2=s_thresh)
	cv2.imshow("FINAL", test)







	# picking out white color
	# 0, 200, 0       255, 255, 255
	lower = np.array([0, 200, 0], dtype="uint8")
	upper = np.array([255, 255, 255], dtype="uint8")
	white_mask = cv2.inRange(hsl, lower, upper)

	# picking out yellow color
	# 80, 0, 100     100, 255, 255
	lower = np.array([80, 0, 100], dtype="uint8")
	upper = np.array([120, 255, 255], dtype="uint8")
	yellow_mask = cv2.inRange(hsl, lower, upper)

	combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
	mask = cv2.bitwise_and(image, image, mask=combined_mask)

	final = cv2.cvtColor(cv2.cvtColor(mask, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(final, (3, 3), 0)
	edge = cv2.Canny(blur, 200, 300)


	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break


cv2.destroyAllWindows()


