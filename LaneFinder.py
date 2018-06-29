import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui as pag
import GameControls as GInput
import math
import traceback

# GLOBAL VARIABLES
drive = True
police_dash = False


class LineList(list):
	def __init__(self):
		list.__init__(self)
		self.size = 0

	def addLine(self, item):
		self.size += 1
		self.append(item)

def lineCountMax(lineCounts):
	return lineCounts[1]

def Lines(image):   # draw HoughLines 
	# 50, 30 for police cam
	# 10, 50 for GTAV
	if police_dash:
		line_min = 40
		gap_max = 50
	else:
		line_min = 50
		gap_max = 50
	lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, np.array([]), line_min, gap_max)

	'''
	# //////////// TESTING
	for line in lines:
		coords = line[0]
		cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), (255,255,255), thickness=2)
	return 0,0
	# //////////////////
	'''

	lineAggregator_L = LineList()  # list to hold the most common lines (LEFT LINE)
	lineAggregator_R = LineList()  # list to hold the most common lines (RIGHT LINE)
	lineCounts_L = LineList()      # list to hold the number of each line (LEFT LINE)
	lineCounts_R = LineList()	   # list to hold the number of each line (RIGHT LINE)

	try:    # add all lines to list and then pick out the 2 strongest lines
		for line in lines:
			coords = line[0]
			PopulateLineLists(coords, lineAggregator_L, lineAggregator_R)

		coord1_a, coord1_b, slope_L = LineFilter(lineAggregator_L, lineCounts_L)
		coord2_a, coord2_b, slope_R = LineFilter(lineAggregator_R, lineCounts_R)

		cv2.line(image, (coord1_a[0], coord1_a[1]), (coord1_b[0], coord1_b[1]), (255,255,255), thickness=2)
		cv2.line(image, (coord2_a[0], coord2_a[1]), (coord2_b[0], coord2_b[1]), (255,255,255), thickness=6)

		return slope_L, slope_R
	except Exception:
		traceback.print_exc()
		print("ERROR: No lines found")
		return 0, 0


def ProcessImage(image, vertices):    # only look at region of interest
	# // SELECTING COLORS // #
	hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	# picking out white color
	lower = np.array([0, 200, 0], dtype="uint8")
	upper = np.array([255, 255, 255], dtype="uint8")
	white_mask = cv2.inRange(hsl, lower, upper)

	# picking out yellow color
	lower = np.array([80, 0, 100], dtype="uint8")
	upper = np.array([120, 255, 255], dtype="uint8")
	yellow_mask = cv2.inRange(hsl, lower, upper)

	combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
	color_mask = cv2.bitwise_and(image, image, mask=combined_mask)

	# // APPLYING EDGE DETECTION // #
	image_grey = cv2.cvtColor(cv2.cvtColor(color_mask, cv2.COLOR_HLS2BGR), cv2.COLOR_BGR2GRAY)       # greyscale
	image_blur = cv2.GaussianBlur(image_grey, (5,5), 0)         # apply blur to improve edges
	image_edges = cv2.Canny(image_blur, 200, 300)               # canny edge

	# // SELECTING REGION // #
	mask = np.zeros_like(image_edges)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(image_edges, mask)

	return masked


def PopulateLineLists(coords, lineAggregator_L, lineAggregator_R):
	dY = -1 * (coords[3] - coords[1])  # inverting since origin point is at top of screen
	dX = coords[2] - coords[0]

	if (dX != 0):  # if denominator is no equal to 0
		print("X1 = {}, Y1 = {} \nX2 = {}, Y2 = {}".format(coords[0], coords[1], coords[2], coords[3]))
		slope = dY / dX
		distance = math.sqrt(math.pow(dX, 2) + math.pow(dY, 2))

		if (slope >= .2):  # POPULATING LEFT LINE
			print("Slope: {}".format(slope))
			y_int = coords[3] + slope * coords[2]
			print("Y-Intercept: {}".format(y_int))

			try:  # add slope and y-intercept to list
				lineAggregator_L.addLine([slope, y_int])
				print("AGGREGATOR SIZE: {}".format(lineAggregator_L.size))
			except:
				print("ERROR: Could not add line")

			# DEBUG printing
			print("dX = {}   dY = {}".format(dX, dY))
			print("This is the slope: {} ".format(dY / dX))
			print("This is distance: {} ".format(distance))

		elif (slope <= -.2):  # POPULATING RIGHT LINE
			print("Slope: {}".format(slope))
			y_int = coords[3] + slope * coords[2]
			print("Y-Intercept: {}".format(y_int))

			try:  # add slope and y-intercept to list
				lineAggregator_R.addLine([slope, y_int])
				print("AGGREGATOR SIZE: {}".format(lineAggregator_R.size))
			# print("AGGREGATOR: {}".format(lineAggregator))
			except:
				print("ERROR: Could not add line")

			# DEBUG printing
			print("dX = {}   dY = {}".format(dX, dY))
			print("This is the slope: {} ".format(dY / dX))
			print("This is distance: {} ".format(distance))
		else:  # if horizontal line
			print("NOT PRINTING HORIZONTAL LINE")
			return 1
	else:  # if denominator is 0
		print("ERROR: dX = 0. Skipping")


def LineFilter(lineAggregator, lineCounts):  # remove unwanted lines
	lineAggregator.sort()
	print(lineAggregator)
	print("THIS IS THE SIZE {}" .format(lineAggregator.size))
	search = True
	i = 0
	while (search):    # populates the repeating lines into lineCount list in the format [index, count]
		print("AGGREGATOR SIZE {} and I = {}" .format(lineAggregator.size, i))
		slope1 = lineAggregator[i][0]
		y_int1 = lineAggregator[i][1]
		count = 1
		j = i + 1
		print("TESTING: I {}" .format(i))
		while (j < lineAggregator.size):
			slope2 = lineAggregator[j][0]
			y_int2 = lineAggregator[j][1]
			if (math.isclose(slope1, slope2, rel_tol=.2) and math.isclose(y_int1, y_int2, rel_tol=.2)):
				count = count + 1
				j = j + 1
				print("COUNT++")
			else:
				lineCounts.append([i, count])
				i = j
				break
		if (j == lineAggregator.size):
			lineCounts.append([i, count])
			search = False
	print("LINE COUNT: {}" .format(lineCounts))

	lineCounts.sort(key=lineCountMax, reverse=True)
	print("SORTED LINE COUNT: {}" .format(lineCounts))

	# calculate the average of the strongest line
	slope_avg1 = 0
	y_int_avg1 = 0
	index_start1 = lineCounts[0][0]
	index_end1 = index_start1 + lineCounts[0][1]
	for i in range(index_start1, index_end1):
		slope_avg1 += lineAggregator[i][0]
		y_int_avg1 += lineAggregator[i][1]
	slope_avg1 /= (index_end1 - index_start1)
	y_int_avg1 /= (index_end1 - index_start1)
	print("SLOPE1: {}       Y-INT: {}" .format(slope_avg1, y_int_avg1))

	#calculating coordinates of the line
	width,height = pag.size()
	print("WIDTH {}    HEIGHT {}" .format(width, height))
	x1_bottom = -(height - y_int_avg1)/slope_avg1
	x1_top = y_int_avg1/slope_avg1
	coord1_a = [int(x1_bottom), int(height)]
	coord1_b = [int(x1_top), 0]

	return coord1_a, coord1_b, slope_avg1





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



def main(): 
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()

	WIDTH,HEIGHT = pag.size()
	vertices = np.array([[0,800], [0,400], [WIDTH/8, 150], [3*WIDTH/8,150], [WIDTH/2,400], [WIDTH/2,800]], np.int32)

	for i in range(0,3):
		print("On the count of 3: {}" .format(i))
		time.sleep(.5)

	while(True):
			# 800x600 windowed mode
			# bbox(x, y, width, height)

			if police_dash == True:
				image = np.array(ImageGrab.grab(bbox=(0, HEIGHT/4, WIDTH/2, 800)))  # grabbing screen into a numpy array  // for police dash
			else:
				image = np.array(ImageGrab.grab(bbox=(0, 0, 900, 600)))  # grabbing screen into a numpy array	// for GTAV

			print("loop took {} seconds " .format(time.time()-last_time))
			last_time = time.time()

			processed_image = ProcessImage(image, [vertices])
			#cv2.imshow("Image", np.hstack([image,image2]))

			slope_L, slope_R = Lines(processed_image)

			if drive:
				pass
				#release_all_controls()
			if (math.fabs(slope_L) > math.fabs(slope_R)):
				cv2.arrowedLine(processed_image, (20, 20), (40, 20), (255, 255, 255), thickness=2, tipLength=.3)
				if drive:
					steer_right()
					#applyGas()
			elif (math.fabs(slope_L) < math.fabs(slope_R)):
				cv2.arrowedLine(processed_image, (40, 20), (20, 20), (255, 255, 255), thickness=2, tipLength=.3)
				if drive:
					steer_left()
					#applyGas()

			cv2.imshow("Image", processed_image)

			# for counting the number of frames
			frames = frames + 1
			print("Number of frames: {} " .format(frames))
			if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					final_time = time.time()
					print("Average FPS: {} " .format(frames/(final_time - elapsed)))
					break


if __name__ == "__main__":
	main()

