import numpy as np
import cv2

train_data = np.load("training_data.npy")
print(len(train_data))
def main():
	for i in range(100000, len(train_data)):
		print(train_data[i][1])

if __name__ == "__main__":
	main()