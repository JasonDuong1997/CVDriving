import os
import numpy as np
import cv2

training_data_file = np.load("training_data.npy")
print("Data length: {}" .format(len(training_data_file)))

data_version = "yuv"
file_name = "training_data_{}.npy" .format(data_version)


def GREY2YUV():
	count = 0

	training_data = []

	for td in training_data_file:
		img = td[0]
		key = td[1]

		bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

		training_data.append([yuv, key])
		print(count)
		count += 1

	print(len(training_data))
	np.save("training_data_yuv.npy", training_data)
	print("Saved!")

def fix():
	img = training_data_file[65999][0]
	bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
	grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	training_data_file[65999][0] = grey

	np.save("training_data.npy", training_data_file)


GREY2YUV()