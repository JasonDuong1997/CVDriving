import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui as pag
import math


class LineList(list):
    def __init__(self):
        list.__init__(self)
        self.size = 0

    def addLine(self, item):
        self.size += 1
        self.append(item)


def Edge(image):    # find edges of the image 
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.Canny(image2, 200, 300)
    image2 = cv2.GaussianBlur(image2, (3,3), 0)
    return image2


def Lines(image):   # draw HoughLines 
    # 70, 30
    line_min = 50
    gap_max = 30
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, np.array([]), line_min, gap_max)

    lineAggregator = LineList()  # list to hold the most common lines
    lineCounts = LineList()      # list to hold the number of each line

    try:
        for line in lines:
            coords = line[0]
            skip = LineFilter(coords, lineAggregator, lineCounts)
            if (skip != 1):
                cv2.line(image, (coords[0],coords[1]), (coords[2],coords[3]), (255,255,255), thickness=2)
    except:
        print("ERROR: No lines found")


def Region(image, vertices):    # only look at region of interest
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked


def LineFilter(coords, lineAggregator, lineCounts):  # remove unwanted lines
    dY = -1*(coords[3] - coords[1])  # inverting since origin point is at top of screen
    dX = coords[2] - coords[0]

    if (dX != 0):   # if denominator is no equal to 0
        print("X1 = {}, Y1 = {} \nX2 = {}, Y2 = {}" .format(coords[0], coords[1], coords[2], coords[3]))
        slope = dY/dX
        distance = math.sqrt(math.pow(dX, 2) + math.pow(dY, 2))

        if (math.fabs(slope) >= .2):    # removing horizontal lines
            print("Slope: {}" .format(slope))
            y_int = coords[3] + slope*coords[1]
            print("Y-Intercept: {}" .format(y_int))

            try:    # add slope and y-intercept to list
                lineAggregator.addLine([slope, y_int])
                print("AGGREGATOR SIZE: {}" .format(lineAggregator.size))
                print("AGGREGATOR: {}" .format(lineAggregator))
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

