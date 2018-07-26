import numpy as np
from PIL import ImageGrab
import cv2
import time
from GameControls import PressKey, ReleaseKey, W, A, D
from KeyLogger import KeyCheck
import win32gui
from AlexNet_Model.TrainData import alexnet
import tensorflow as tf


# TESTING
WIDTH = 80
HEIGHT = 62

LR = 1e-3		# learning rate
EPOCHS = 8
MODEL_NAME = "pygta5-SmartCar-{}-{}-{}-epochs.model" .format(LR, "alexnet", EPOCHS)

def straight():
	ReleaseKey(A)
	ReleaseKey(D)
	PressKey(W)

def left():
	ReleaseKey(W)
	ReleaseKey(D)
	PressKey(A)

def right():
	ReleaseKey(A)
	ReleaseKey(W)
	PressKey(D)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

train_data = np.load("training_data.npy")

def main():

	for i in range(0,3):
		print("On the count of 3: {}" .format(i))
		time.sleep(.5)

	for td in train_data:
		cv2.imshow("TEST", td[0])
		prediction = model.predict(td[0].reshape(1, WIDTH, HEIGHT, 1))[0]
		moves = list(np.around(prediction))
		print("Move   {}" .format(moves))
		print("Actual {}" .format(td[1]))

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break



if __name__ == "__main__":
	main()