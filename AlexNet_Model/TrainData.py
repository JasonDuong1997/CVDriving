# using pre-made AlexNet

from __future__ import division, print_function, absolute_import
import win32gui
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def alexnet(width, height, lr, output=3):
	network = input_data(shape=[None, width, height, 1], name='input')
	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, output, activation='softmax')
	network = regression(network, optimizer='momentum',
						 loss='categorical_crossentropy',
						 learning_rate=lr, name='targets')

	model = tflearn.DNN(network, checkpoint_path='model_alexnet',
						max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir='log')

	return model


hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
win_x = rect[0]
win_y = rect[1]
win_w = rect[2] - win_x
win_h = rect[3] - win_y

WIDTH = int(win_w/10)
HEIGHT = int(win_h/10)
LR = 1e-3		# learning rate
EPOCHS = 8
MODEL_NAME = "pygta5-SmartCar-{}-{}-{}-epochs.model" .format(LR, "alexnet", EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load("training_data_balanced.npy")

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_Y = [i[1] for i in test]

model.fit({"input": X}, {"targets": Y}, n_epoch=EPOCHS, validation_set=({"input": test_X}, {"targets": test_Y}),
		  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# tensorboard --logdir=foo:C:\Users\xSoapBubble\PycharmProjects\CVDriving

model.save(MODEL_NAME)
