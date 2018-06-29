import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui as pag


while (1):
	WIDTH,HEIGHT = pag.size()
	image = np.array(ImageGrab.grab(bbox=(0, HEIGHT / 4, WIDTH / 2, 800)))
	hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

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

	cv2.imshow("TEST", edge)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()


