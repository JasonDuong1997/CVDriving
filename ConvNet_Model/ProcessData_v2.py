import cv2
import numpy as np
import pandas as pd
from collections import Counter


def resize(src_img):
	return cv2.resize(src_img, dsize=(160,120))


def shuffle(training_data):
	print("before shuffle {}" .format(len(training_data)))
	np.random.shuffle(training_data)
	print("after shuffle {}" .format(len(training_data)))

	return training_data


def main():
	training_data = np.load("udacity_training_data.npy", encoding="latin1")

	# shrinking images
	for item in training_data:
		item[0] = resize(item[0])
		item[1] = [float(item[1])]

	# randomizing order of training data
	processed_data = shuffle(training_data)

	np.save("udacity_trainingData_processed2.npy", processed_data)


main()