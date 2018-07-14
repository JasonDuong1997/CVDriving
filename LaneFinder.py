import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui as pag
import GameControls as GInput
import win32gui
import math
import traceback

# GLOBAL VARIABLES
drive = False
police_dash = False

hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[1]
win_w = rect[2] - win_x
win_h = rect[3] - win_y

class LineList(list):
	def __init__(self):
		list.__init__(self)
		self.size = 0

	def addLine(self, item):
		self.size += 1
		self.append(item)

class LineAverager():
	def __init__(self):
		self.size = 0
		self.coordA = []
		self.coordB = []
		self.slope = []
		self.y_int = []
		self.currAvg = []

	def addLine(self, coordA_n, coordB_n, slope_n, y_int_n):
		self.size += 1
		self.coordA.append(coordA_n)
		self.coordB.append(coordB_n)
		self.slope.append(slope_n)
		self.y_int.append(y_int_n)

	def average(self):
		try:
			# TODO: use weighted average
			print("Calculating AVERAGE")
			coordA0_sum = 0
			coordA1_sum = 0
			coordB0_sum = 0
			coordB1_sum = 0
			for cA in self.coordA:
				coordA0_sum += cA[0]
				coordA1_sum += cA[1]
			for cB in self.coordB:
				coordB0_sum += cB[0]
				coordB1_sum += cB[1]
			coordA_avg = [int(coordA0_sum/self.size), int(coordA1_sum/self.size)]
			coordB_avg = [int(coordB0_sum/self.size), int(coordB1_sum/self.size)]

			slope_sum = 0
			for s in self.slope:
				slope_sum += s
			slope_avg = slope_sum/self.size

			y_int_sum = 0
			for y in self.y_int:
				y_int_sum += y
			y_int_avg = y_int_sum/self.size

			self.currAvg = [coordA_avg, coordB_avg, slope_avg, y_int_avg]
			return coordA_avg, coordB_avg, slope_avg, y_int_avg
		except:
			print("ERROR: Could not calculate the average line.")
			return [0,0], [0,0], 0, 0

	def clearLists(self):
		for i in range(0, self.size):
			del self.coordA[0]
			del self.coordB[0]
			del self.slope[0]
			del self.y_int[0]
		self.size = 0


def lineCountMax(lineCounts):
	return lineCounts[1]

def Lines(image, frames, avgLine_L, avgLine_R):   # draw HoughLines
	# 50, 30 for police cam
	# 10, 50 for GTAV
	if police_dash:
		line_min = 40
		gap_max = 50
	else:
		line_min = 10
		gap_max = 60
	lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, np.array([]), line_min, gap_max)
	lineAggregator_L = LineList()  # list to hold the most common lines (LEFT LINE)
	lineAggregator_R = LineList()  # list to hold the most common lines (RIGHT LINE)
	lineCounts_L = LineList()      # list to hold the number of each line (LEFT LINE)
	lineCounts_R = LineList()	   # list to hold the number of each line (RIGHT LINE)

	renderCycle = 1  # number of frames to grab line averages over

	try:    # add all lines to list and then pick out the 2 strongest lines
		for line in lines:
			coords = line[0]
			PopulateLineLists(coords, lineAggregator_L, lineAggregator_R)

		coord1_a, coord1_b, slope_L, y_int_L = LineFilter(lineAggregator_L, lineCounts_L)
		coord2_a, coord2_b, slope_R, y_int_R = LineFilter(lineAggregator_R, lineCounts_R)

		avgLine_L.addLine(coord1_a, coord1_b, slope_L, y_int_L)
		avgLine_R.addLine(coord2_a, coord2_b, slope_R, y_int_R)

		if (frames % renderCycle == 0):
			avgLine_L.average()
			avgLine_R.average()
			avgLine_L.clearLists()
			avgLine_R.clearLists()
		if (frames >= renderCycle):
			# updating ROI
			x_van,y_van = FindVanishingPoint(avgLine_L.currAvg[2], avgLine_R.currAvg[2], avgLine_L.currAvg[3], avgLine_R.currAvg[3])
			cv2.circle(image, (int(x_van), int(y_van)), 10, (255,255,255), thickness=3)

			cv2.line(image, (avgLine_L.currAvg[0][0], avgLine_L.currAvg[0][1]), (int(x_van), int(y_van)), (255,255,255), thickness=2)
			cv2.line(image, (avgLine_R.currAvg[0][0], avgLine_R.currAvg[0][1]), (int(x_van), int(y_van)), (255,255,255), thickness=6)
		else:
			# updating ROI
			x_van,y_van = FindVanishingPoint(slope_L, slope_R, y_int_L, y_int_R)
			cv2.circle(image, (int(x_van), int(y_van)), 10, (255,255,255), thickness=3)

			cv2.line(image, (coord1_a[0], coord1_a[1]), (int(x_van), int(y_van)), (255,255,255), thickness=2)
			cv2.line(image, (coord2_a[0], coord2_a[1]), (int(x_van), int(y_van)), (255,255,255), thickness=6)

		return avgLine_L.currAvg[2], avgLine_R.currAvg[2], y_van
	except Exception:
		traceback.print_exc()
		print("ERROR: No lines found")
		return 0, 0, 175


def ProcessImage(image, vertices):    # only look at region of interest
	# // COLOR SELECTION // #
	"""
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
	"""
	# selecting well-lit white lines
	red = image[:,:,0]
	green = image[:,:,1]
	red_threshed = cv2.inRange(red, 210, 255)
	green_threshed = cv2.inRange(green, 200, 255)
	white_lit = cv2.bitwise_and(src1=red_threshed, src2=green_threshed)

	# selecting well-lit white lines (testing)
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l = lab[:,:,0]
	l_threshed = cv2.inRange(l, 200, 255)
	white_lit = l_threshed

	# selecting well-lit yellow lines
	hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	s = hsl[:,:,2]
	yellow_lit = cv2.inRange(s, 170, 255)

	# checking for any lines in the shadows
	shadow_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	shadow_blur = cv2.GaussianBlur(shadow_grey, (5,5), 0)         									 # apply blur to smooth
	shadow_preedges = cv2.Canny(shadow_blur, threshold1=100, threshold2=170)
	l = hsl[:,:,1]
	l_threshed = cv2.inRange(l, 70, 100)
	white_shadow = shadow_preedges & l_threshed

	# selecting shadowed yellow lines
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	h = hsv[:,:,0]
	sv = hsv[:,:,1]
	sv_threshed = cv2.inRange(sv, 115, 255)
	h_threshed = cv2.inRange(h, 80, 100)
	yellow_shadow = sv_threshed & h_threshed

	# adding up all of the thresholds
	color_threshed = white_lit | yellow_lit | yellow_shadow
	color_mask = cv2.bitwise_and(image, image, mask=color_threshed)

	# // APPLYING EDGE DETECTION // #
	image_grey = cv2.cvtColor(cv2.cvtColor(color_mask, cv2.COLOR_HLS2BGR), cv2.COLOR_BGR2GRAY)       # grey-scale
	image_blur = cv2.GaussianBlur(image_grey, (5,5), 0)         									 # apply blur to smooth
	image_edges = cv2.Canny(image_blur, threshold1=200, threshold2=300)              				 # Canny edge

	total_edges = image_edges #| shadow_edges

	# // SELECTING REGION // #
	mask = np.zeros_like(total_edges)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(total_edges, mask)

	cv2.imshow("Processed_Image", masked)
	return masked


def FindVanishingPoint(slope1, slope2, y_int1, y_int2):
	x = -(y_int2 - y_int1)/(slope1 - slope2)
	y = slope1*(-x) + y_int1

	return x,y



def PopulateLineLists(coords, lineAggregator_L, lineAggregator_R):
	dY = -1 * (coords[3] - coords[1])  # inverting since origin point is at top of screen
	dX = coords[2] - coords[0]

	s_threshold = .2

	if (dX != 0):  # if denominator is no equal to 0
		print("X1 = {}, Y1 = {} \nX2 = {}, Y2 = {}".format(coords[0], coords[1], coords[2], coords[3]))
		slope = dY / dX
		distance = math.sqrt(math.pow(dX, 2) + math.pow(dY, 2))

		if (slope >= s_threshold):  # POPULATING LEFT LINE
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

		elif (slope <= -s_threshold ):  # POPULATING RIGHT LINE
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
	slope_avg = 0
	y_int_avg = 0
	index_start = lineCounts[0][0]
	index_end = index_start + lineCounts[0][1]
	for i in range(index_start, index_end):
		slope_avg += lineAggregator[i][0]
		y_int_avg += lineAggregator[i][1]
	slope_avg /= (index_end - index_start)
	y_int_avg /= (index_end - index_start)
	print("SLOPE1: {}       Y-INT: {}" .format(slope_avg, y_int_avg))

	# checking 2nd strongest line for closest x-intercept
	if (len(lineCounts) > 1):
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

	#calculating coordinates of the line
	x_bottom1 = -(win_h - y_int_avg)/slope_avg
	if (len(lineCounts) > 1):
		x_bottom2 = -(win_h - y_int_avg2)/slope_avg2

		# return line that is closest to vertical midline of the screen
		if (math.fabs(win_w/2 - x_bottom1) < math.fabs(win_w/2 - x_bottom2)):
			coord_a = [int(x_bottom1), int(win_h)]
			x_top1 = y_int_avg/slope_avg
			coord_b = [int(x_top1), 0]
		else:
			coord_a = [int(x_bottom2), int(win_h)]
			x_top2 = y_int_avg2/slope_avg2
			coord_b = [int(x_top2), 0]
			slope_avg = slope_avg2
			y_int_avg = y_int_avg2
	else:
		coord_a = [int(x_bottom1), int(win_h)]
		x_top1 = y_int_avg / slope_avg
		coord_b = [int(x_top1), 0]

	return coord_a, coord_b, slope_avg, y_int_avg


def main(): 
	frames = 0.0
	elapsed = time.time()
	last_time = time.time()
	frame_loop = 0

	WIDTH,HEIGHT = pag.size()
	#vertices = np.array([[0,800], [0,350], [WIDTH/8, y_van], [3*WIDTH/8,y_van], [WIDTH/2,350], [WIDTH/2,800],  [9*WIDTH/32, 350], [7*WIDTH/32, 350]], np.int32)

	y_van = 11*win_h/50

	avgLine_L = LineAverager()
	avgLine_R = LineAverager()

	for i in range(0,3):
		print("On the count of 3: {}" .format(i))
		time.sleep(.5)

	while(True):
			# 800x600 windowed mode
			# bbox(x, y, width, height)
			vertices = np.array([[0, win_h], [0, 7*win_h/16], [win_w/4, y_van], [3*win_w/4, y_van], [win_w, 7*win_h/16], [win_w, win_h], [5*win_w/8, 11*win_h/16], [3*win_w/8, 11*win_h/16]], np.int32)

			if police_dash == True:
				image = np.array(ImageGrab.grab(bbox=(0, HEIGHT/4, WIDTH/2, 800)))  # grabbing screen into a numpy array  // for police dash
			else:
				image = np.array(ImageGrab.grab(bbox=(win_x, win_y, rect[2], rect[3])))  # grabbing screen into a numpy array	// for GTAV


			print("loop took {} seconds " .format(time.time()-last_time))
			last_time = time.time()

			processed_image = ProcessImage(image, [vertices])
			#cv2.imshow("Image", np.hstack([image,image2]))

			slope_L, slope_R, y_van = Lines(processed_image, frames, avgLine_L, avgLine_R)

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

