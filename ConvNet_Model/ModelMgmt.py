import tensorflow as tf

def model_save():
	x = tf.Variable(tf.random_normal(shape=[2]), name="x")
	y = tf.Variable(tf.random_normal(shape=[3]), name="y")

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver([x])
		saver.save(sess, "./test_model")
		print(x.eval())
		print(y.eval())

def model_load():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver = tf.train.import_meta_graph("./test_model.meta")
		saver.restore(sess, tf.train.latest_checkpoint("./"))
		print(sess.run("x:0"))
		print(sess.run("y:0"))
