export:
	roslaunch export.launch bagfile:="HMB_1_images.bag" _image_transport:=compressed

filter:
	rosbag filter ${IN_BAG} ${OUT_BAG} "topic == '/center_camera/image_color/compressed' or topic == '/vehicle/steering_report'"

republish:
	rosrun image_transport republish compressed in:=/center_camera/image_color _image_transport:=compressed raw out:=/center_camera/image_color/raw _use_sim_time:=True

play:
	rosbag play HMB_${BAG_NUM}_filtered.bag --clock

record:
	rosbag record -O HMB_${BAG_NUM}_data_raw /center_camera/image_color/raw /vehicle/steering_report _use_sim_time:=True	

changeExport:
	cp export.launch.txt export.launch

changeRepublish:
	cp republish.launch.txt republish.launch

make rqt_bag:
	rqt_bag ${BAG_FILE}  _use_sim_time:=true
