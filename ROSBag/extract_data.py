"""
*** NOTE: This file only works on Linux because the ROS API is only available on Linux OS ***
	- I used Windows 10 WSL to install a Linux distribution on my Windows machine 
	- Used on Udacity Self-Driving Dataset
"""

import rosbag
import numpy as np


bag_number = "6"
bagfile = "HMB_{}_data_raw.bag" .format(bag_number)
data_filename = "HMB_{}_extracted.npy" .format(bag_number)

# returns the value from a msg topic 
def subject_filter(input_str, subject):
	subject_str = subject + ": "
	str0 = input_str.split(subject_str)
	str1 = str0[1].split("\n", 1)
	return str1[0]

# extracts the list of times correlated to each image
def extract_imageList(bag):
	image_list = []

	count = 0
	
	print("Extracting Images...")
	for topic, msg, t in bag:
		if (topic == '/center_camera/image_color/raw'):
			# list of [secs, nsecs, imageData]
			msg_str = str(msg)
			secs = subject_filter(msg_str, "secs")
			nsecs = int(subject_filter(msg_str, "nsecs"))
			img_data = extract_imageData(msg)			
			
			image_list.append([secs, nsecs, img_data])
	
			count += 1
			print("Count {}" .format(count))
	
	return image_list

	
# extracts steering angle synchronized to time
def extract_steering(bag):
	steering_list = []
	
	print("Extracting Steering...")
	for topic, msg, t in bag:
		if (topic == '/vehicle/steering_report'):
			# list of [secs, nsecs, steering wheel angle]
			msg_str = str(msg)
			secs = subject_filter(msg_str, "secs")
			nsecs = int(subject_filter(msg_str, "nsecs"))	
			steering_data = subject_filter(msg_str, "steering_wheel_angle")		
		
			steering_list.append([secs, nsecs, steering_data])
			
	return steering_list


# extracts actual image data from msg
def extract_imageData(msg):
	str0 = str(msg).split("data: ")
	str1 = str0[1].split("\n", 1)

	stripped = (str1[0].rstrip("]")).lstrip("[")
	stripped = stripped.split(", ")
	
	pixelList = []
	for pixel in stripped:
		pixelList.append(int(pixel))
	
	imageBuff = np.array(pixelList, dtype=np.uint8)
	imageData = imageBuff.reshape(480,640,3)
	return imageData

	
def main():		
	bag = rosbag.Bag(bagfile)

	# extracting data
	image_list = extract_imageList(bag)
	steering_list = extract_steering(bag)

	final_data = []
	
	count = 0
	# matching times
	print("Matching times...")
	for i in range(len(image_list)):
		sec_key = image_list[i][0]	#string
		nsec_key = image_list[i][1]	#int
		print("Searching match for: {}" .format(sec_key))
		for j in range(len(steering_list)):
			if (sec_key == steering_list[j][0]):
				if(steering_list[j][1] >= nsec_key):
					final_data.append([image_list[i][2], float(steering_list[j][2])])
					count += 1
					print("Match Count: {}" .format(count))	
					break
		

	print("Length of final_data: {}" .format(len(final_data)))
	np.save(data_filename, final_data)	


main()
