import pandas as pd
from collections import Counter
from random import shuffle
import numpy as np

data_version = "yuv"
train_data = np.load("training_data_{}.npy" .format(data_version))

def full_shuffle():
	df = pd.DataFrame(train_data)
	print(df.head())
	print(Counter(df[1].apply(str)))

	lefts = []
	rights = []
	forwards = []

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
	max_count = min(counts)

	lefts = lefts[:max_count]
	forwards = forwards[:max_count]
	rights = rights[:max_count]

	total_data = lefts + forwards + rights
	print(len(total_data))

	shuffle(total_data)
	np.save("training_data_{}_balanced.npy" .format(data_version), total_data)

	"""
	forwards = forwards[:len(lefts)][:len(rights)]
	lefts = lefts[:len(forwards)]
	rights = rights[:len(rights)]
	
	final_data = forwards + lefts + rights
	print(len(final_data))
	
	np.save("training_data_balanced.npy", final_data)
	"""

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

partial_shuffle()