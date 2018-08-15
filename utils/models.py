import tensorflow as tf
import numpy as np


class BaseModel(object):

	def __init__(self):
		super(BaseModel, self).__init__()
		# Remember to intialize Saver!

	def save(self, sess, savepath, global_step=None, prefix="ckpt", verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(sess, savepath + prefix, global_step=global_step)
		if verbose:
			print("Model saved to {}.".format(savepath + prefix + '-' + str(global_step)))

	def load(self, sess, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(sess, ckpt)
		if verbose:
			print("Model loaded from {}.".format(ckpt))

class CNN(BaseModel):

	def __init__(self, name, inputs, output_dim, parent):
		super(CNN, self).__init__()
		self.name = name
		self.inputs = inputs
		self.output_dim = output_dim
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent + '/' + self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):
		conv_1 = tf.layers.conv2d(
			inputs=self.inputs,
			filters=16,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=tf.nn.relu,
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_1',
		)
		conv_2 = tf.layers.conv2d(
			inputs=conv_1,
			filters=32,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=tf.nn.relu,
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_2',
		)
		unroll = tf.reshape(conv_2, [-1, 11*11*32])
		dense = tf.layers.dense(
			inputs=unroll,
			units=self.output_dim,
			activation=None,
			name='dense',
		)
		self.outputs = dense

class FFN(BaseModel):

	def __init__(self, name, inputs, output_dim, parent):
		super(FFN, self).__init__()
		self.name = name
		self.inputs = inputs
		self.output_dim = output_dim
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent + '/' + self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):
		dense_1 = tf.layers.dense(
			inputs=self.inputs,
			units=64,
			activation=tf.nn.relu,
			name='dense_1',
		)
		dense_2 = tf.layers.dense(
			inputs=dense_1,
			units=self.output_dim,
			activation=None,
			name='dense_2',
		)
		self.outputs = dense_2
