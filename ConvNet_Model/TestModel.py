import numpy as np
import tensorflow as tf
import cv2

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.import_meta_graph("./CNN_Model.meta")
	saver.restore(sess, tf.train.latest_checkpoint("./"))
	print(sess.run("B_conv1:0"))