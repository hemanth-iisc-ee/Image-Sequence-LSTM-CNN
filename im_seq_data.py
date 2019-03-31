from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import pdb


class ImageSeqGenerator():
	def __init__(self):
		# load data
		(X_train, y_train), (X_test, y_test) = mnist.load_data()

		self.imh = X_train.shape[1]
		self.imw = X_train.shape[2]
		self.imch = 1

		# reshape to be [samples][pixels][width][height]
		X_train = X_train.reshape(X_train.shape[0], self.imch, self.imh, self.imw).astype('float32')
		X_test = X_test.reshape(X_test.shape[0], self.imch, self.imh, self.imw).astype('float32')

		self.X_train = X_train / 255
		self.X_test = X_test / 255

		# one hot encode outputs
		self.y_train = np_utils.to_categorical(y_train)
		self.y_test = np_utils.to_categorical(y_test)

		self.y_train_raw = y_train
		self.y_test_raw = y_test
		

		self.alphabet = np.unique(y_train)

		return

	def mnist_train_data(self):
		return self.X_train, self.y_train

	def mnist_test_data(self):
		return self.X_test, self.y_test

	def __sample_pos_idx(self,seq_len, y_raw):
		a_label = np.random.choice(self.alphabet)
		sample_idx = np.random.choice(np.flatnonzero(y_raw  == a_label),size=seq_len)
		return sample_idx

	def __sample_neg_idx(self, seq_len, y_raw):

		while True: 
			labels = np.random.choice(self.alphabet,seq_len)
			if len(np.unique(labels)) > 1:
				break

		sample_idx = []
		for l in labels:
			sample_idx.append(np.random.choice(np.flatnonzero(y_raw  == l),size=1))
		return np.array(sample_idx)[:,0]

			
	def gen_train_sequences(self, seq_len=3, n_samples=1, pos_class_prob = 0.5):

		sample_idx = np.array([]).astype(int)
		y = np.zeros(n_samples)
		for i in range(n_samples):
			if np.random.uniform() < pos_class_prob:
				a_seq_idx = self.__sample_pos_idx(seq_len, self.y_train_raw)
				y[i] = 1
			else:
				a_seq_idx = self.__sample_neg_idx(seq_len, self.y_train_raw)

			sample_idx = np.concatenate((sample_idx , a_seq_idx))

		X_seq = self.X_train[sample_idx,:,:,:].copy()

		X_seq = X_seq.reshape(n_samples,seq_len, self.imch, self.imh, self.imw)

		return X_seq, np_utils.to_categorical(y)

	def gen_test_sequences(self, seq_len=3, n_samples=1, pos_class_prob = 0.5):

		sample_idx = np.array([]).astype(int)
		y = np.zeros(n_samples)
		for i in range(n_samples):
			if np.random.uniform() < pos_class_prob:
				a_seq_idx = self.__sample_pos_idx(seq_len, self.y_test_raw)
				y[i] = 1
			else:
				a_seq_idx = self.__sample_neg_idx(seq_len, self.y_test_raw)
			sample_idx = np.concatenate((sample_idx , a_seq_idx))


		X_seq = self.X_test[sample_idx,:,:,:].copy()

		X_seq = X_seq.reshape(n_samples,seq_len, self.imch, self.imh, self.imw)

		return X_seq, np_utils.to_categorical(y)






