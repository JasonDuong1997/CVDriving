import os
import numpy as np
import cv2

training_data_file = np.load("training_data.npy")
print("Data length: {}" .format(len(training_data_file)))

data_version = "yuv"
file_name = "training_data_{}.npy" .format(data_version)

if os.path.isfile(file_name):
	print("File exists,. Loading previous data.")
	training_data = list(np.load(file_name))
else:
	print("File does not exist. Starting fresh.")
	training_data = []

def GREY2YUV():
	for td in training_data_file:
		img = td[0]
		key = td[1]

		bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

		training_data.append([yuv, key])

	print(len(training_data))
	np.save(file_name, training_data)

GREY2YUV()