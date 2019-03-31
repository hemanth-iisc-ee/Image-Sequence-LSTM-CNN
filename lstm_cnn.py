from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.models import Model
from keras import backend as K
from keras.utils import np_utils
K.set_image_dim_ordering('th')






class CnnLstmModel():
	def __init__(self):
		# load data
		(X_train, y_train), (X_test, y_test) = mnist.load_data()

		self.num_classes = np.unique(y_train).shape[0]
		self.imh = X_train.shape[1]
		self.imw = X_train.shape[2]
		self.imch = 1

		return

	def cnn(self, model_file=None, output_layer=True, trainable=True):
		# create model
		model = Sequential()
		model.add(Conv2D(32, (5, 5), input_shape=(self.imch, self.imh, self.imw), activation='relu',name='conv'))
		model.add(MaxPooling2D(pool_size=(2, 2),name='pool'))
		model.add(Dropout(0.2,name='dropout'))
		model.add(Flatten())
		model.add(Dense(128, activation='relu',name='dense'))
		if output_layer:
			model.add(Dense(self.num_classes, activation='softmax',name='output'))
		else:
			print("CNN dropped softmax layer")

		if model_file:
			print("CNN load_weights")
			model.load_weights(model_file, by_name=True)

		if not trainable:
			print("CNN Making layers non Trainable")
			for layer in model.layers:
				layer.trainable = False

		return model

	def lstm_cnn(self, model_file=None, MAX_SEQ_LEN=3, cnn_model_file='cnn_model.h5'):

		cnn = self.cnn(model_file=cnn_model_file, output_layer=False, trainable=False)

		input_layer = Input(shape=(MAX_SEQ_LEN, self.imch, self.imh, self.imw))

		lstm_ip_layer = TimeDistributed(cnn)(input_layer)

		lstm = LSTM(128)(lstm_ip_layer)
		output = Dense(units=2,activation='softmax')(lstm)

		model = Model([input_layer],output)
		if model_file:
			model.load_weights(model_file)

		return model

