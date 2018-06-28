import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui as pag
import math
import traceback

class LineList(list):
	def __init__(self):
		list.__init__(self)
		self.size = 0

	def addLine(self, item):
		self.size += 1
		self.append(item)


def Edge(image):    # find edges of the image
	image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # greyscale
	image2 = cv2.Canny(image2, 200, 300)                # canny edge
	image2 = cv2.GaussianBlur(image2, (3,3), 0)         # apply blur to improve edges
	return image2


def Lines(image):   # draw HoughLines 
	# 70, 30
	line_min = 50
	gap_max = 30
	lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, np.array([]), line_min, gap_max)

	lineAggregator = LineList()  # list to hold the most common lines
	lineCounts = LineList()      # list to hold the number of each line

	try:    # add all lines to list and then pick out the 2 strongest lines
		for line in lines:
			coords = line[0]
			PopulateLineLists(coords, lineAggregator)

		coord1_a, coord1_b, coord2_a, coord2_b = LineFilter(lineAggregator, lineCounts)
		print(coord1_a)
		print(coord1_b)
		print(coord2_a[1])
		print(coord2_b[1])
		cv2.line(image, (coord1_a[0], coord1_a[1]), (coord1_b[0], coord1_b[1]), (255,255,255), thickness=2)
		cv2.line(image, (coord2_a[0], coord2_a[1]), (coord2_b[0], coord2_b[1]), (255,255,255), thickness=2)
		'''if (skip != 1):
			cv2.line(image, (coords[0],coords[1]), (coords[2],coords[3]), (255,255,255), thickness=2)'''
	except Exception:
		traceback.print_exc()
		print("ERROR: No lines found")


def Region(image, vertices):    # only look at region of interest
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(image, mask)
	return masked


def PopulateLineLists(coords, lineAggregator):
	dY = -1 * (coords[3] - coords[1])  # inverting since origin point is at top of screen
	dX = coords[2] - coords[0]

	if (dX != 0):  # if denominator is no equal to 0
		print("X1 = {}, Y1 = {} \nX2 = {}, Y2 = {}".format(coords[0], coords[1], coords[2], coords[3]))
		slope = dY / dX
		distance = math.sqrt(math.pow(dX, 2) + math.pow(dY, 2))

		if (math.fabs(slope) >= .2):  # removing horizontal lines
			print("Slope: {}".format(slope))
			y_int = coords[3] + slope * coords[2]
			print("Y-Intercept: {}".format(y_int))

			try:  # add slope and y-intercept to list
				lineAggregator.addLine([slope, y_int])
				print("AGGREGATOR SIZE: {}".format(lineAggregator.size))
			   # print("AGGREGATOR: {}".format(lineAggregator))
			except:
				print("ERROR: Could not add line")

			# DEBUG printing
			print("dX = {}   dY = {}".format(dX, dY))
			print("This is the slope: {} ".format(dY / dX))
			print("This is distance: {} ".format(distance))
		else:
			print("NOT PRINTING HORIZONTAL LINE")
			return 1
	else:
		print("ERROR: dX = 0. Skipping")


def LineFilter(lineAggregator, lineCounts):  # remove unwanted lines
	lineAggregator.sort()
	print(lineAggregator)
	print("THIS IS THE SIZE {}" .format(lineAggregator.size))
	search = True
	i = 0
	while (search):    # populates the repeating lines into lineCount list in the format [index, count]
		slope1 = lineAggregator[i][0]
		y_int1 = lineAggregator[i][1]
		count = 1
		j = i + 1
		print("TESTING: I {}" .format(i))
	 #   print("TESTING: J {}" .format(j))
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

	# calculate the average of the top 2 strongest lines
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

	slope_avg2 = 0
	y_int_avg2 = 0
	index_start2 = lineCounts[1][0]
	index_end2 = index_start2 + lineCounts[1][1]
	for i in range(index_start2, index_end2):
		slope_avg2 += lineAggregator[i][0]
		y_int_avg2 += lineAggregator[i][1]
	slope_avg2 /= (index_end2 - index_start2)
	y_int_avg2 /= (index_end2 - index_start2)
	print("SLOPE2: {}       Y-INT: {}" .format(slope_avg2, y_int_avg2))

	#calculating coordinates of the two lines
	width,height = pag.size()
	print("WIDTH {}    HEIGHT {}" .format(width, height))
	x1_bottom = -(height - y_int_avg1)/slope_avg1
	x1_top = y_int_avg1/slope_avg1
	coord1_a = [int(x1_bottom), int(height)]
	coord1_b = [int(x1_top), 0]

	x2_bottom = -(height - y_int_avg2)/slope_avg2
	x2_top = y_int_avg2/slope_avg2
	coord2_a = [int(x2_bottom), int(height)]
	coord2_b = [int(x2_top), int(0)]

	return coord1_a, coord1_b, coord2_a, coord2_b


def lineCountMax(lineCounts):
	return lineCounts[1]

def main(): 
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()

	WIDTH,HEIGHT = pag.size()
	vertices = np.array([[0,800], [0,400], [WIDTH/8, 150], [3*WIDTH/8,150], [WIDTH/2,400], [WIDTH/2,800]], np.int32)

	while(True):
			# 800x600 windowed mode
			# bbox(x, y, width, height)
			image = np.array(ImageGrab.grab(bbox=(0, HEIGHT/4, WIDTH/2, 800)))  # grabbing screen into a numpy array

			print("loop took {} seconds " .format(time.time()-last_time))
			last_time = time.time()

			image2 = Edge(image)
			image2 = Region(image2, [vertices])
			Lines(image2)

			cv2.imshow("Image", image2)

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

