import numpy as np
import cv2

train_data = np.load("training_data.npy")
print(len(train_data))
def main():
	for td in train_data:
		cv2.imshow("image", td[0])
		print(td[1])


		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

if __name__ == "__main__":
	main()