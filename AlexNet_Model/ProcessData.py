import pandas as pd
from collections import Counter
from random import shuffle
import numpy as np

train_data = np.load("training_data.npy")

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)
for data in train_data:
	image = data[0]
	choice = data[1]

	if (choice == [1,0,0]):
		lefts.append([image, choice])
	elif (choice == [0,1,0]):
		forwards.append([image, choice])
	elif (choice == [0,0,1]):
		rights.append([image, choice])

forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(rights)]

final_data = forwards + lefts + rights
print(len(final_data))

np.save("training_data_balanced.npy", final_data)