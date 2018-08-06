import pandas as pd
from collections import Counter
from random import shuffle
import numpy as np
import tensorflow as tf

data_version = "yuv"
train_data = np.load("training_data_yuv_balanced.npy")

df = pd.DataFrame(train_data)
print(Counter(df[1].apply(str)))

def full_shuffle():
	lefts = []
	rights = []
	forwards = []

	print("before shuffle {}" .format(len(train_data)))
	np.random.shuffle(train_data)
	print("after shuffle {}" .format(len(train_data)))

	for data in train_data:
		image = data[0]
		choice = data[1]

		if (choice == [1,0,0]):
			lefts.append([image, choice])
		elif (choice == [0,1,0]):
			forwards.append([image, choice])
		elif (choice == [0,0,1]):
			rights.append([image, choice])

	counts = [len(lefts), len(forwards), len(rights)]
	print(counts)
	min_count = min(counts)

	lefts = lefts[:min_count]
	forwards = forwards[:int(min_count*1.8)]
	rights = rights[:max(int(min_count*1.2), len(rights))]

	total_data = lefts + forwards + rights
	print(len(total_data))

	shuffle(total_data)
	np.save("training_data_yuv_balanced", total_data)

def partial_shuffle():
	size_of_data = len(train_data)

	test_size = int(0.03*size_of_data)

	test_data = train_data[-test_size:]
	np.save("test_data_yuv.npy", test_data)
	print("Test data saved!")

	training_data = train_data[:-test_size]
	shuffle(training_data)
	np.save("training_data_shuffled.npy", training_data)
	print("Training data saved!")

