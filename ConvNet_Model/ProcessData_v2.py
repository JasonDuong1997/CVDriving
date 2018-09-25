import time
import cv2
import numpy as np
import pandas as pd
from collections import Counter

###  DATA PROCESSING PIPELINE ###
# 1. raw images    & angles in string values
# 2. resize images & convert angles to float values
# 3. ------------- & multiply angles by -1
# 4. double data by saving h-flipping images and angles multiplied by -1 again


def resize(src_img, width, height):
	return cv2.resize(src_img, dsize=(width, height))

def shuffle(training_data):
	print("before shuffle {}" .format(len(training_data)))
	np.random.shuffle(training_data)
	print("after shuffle {}" .format(len(training_data)))

	return training_data


def h_flip(src_img):
	return cv2.flip(src_img, 1)

def v_flip(src_img):
	return cv2.flip(src_img, 0)


def main():
	print("Loading data...")
	training_data = np.load("udacity_training_data.npy", encoding="latin1")

	h_flip_training_data = []

	print("Processing data...")
	for item in training_data:
		item[0] = resize(item[0], width=80, height=60)	# shrinking images
		item[1] = [float(item[1])*-1]					# converting angle values from string to float

		h_flip_training_data.append([h_flip(item[0]), [item[1]*-1]])	# duplicating data and then horizontally flipping

	print(len(training_data))
	training_data = training_data + h_flip_training_data

	# randomizing order of training data
	processed_data = shuffle(training_data)

	print(len(processed_data))

	print("Saving data...")
	np.save("udacity_trainingData_processed.npy", processed_data)
