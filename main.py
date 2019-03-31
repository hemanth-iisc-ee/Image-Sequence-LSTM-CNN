from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from im_seq_data import ImageSeqGenerator 
from lstm_cnn import CnnLstmModel  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)


# Model creator
create_model =  CnnLstmModel()

# Model files
cnn_model_file = 'cnn_model.h5'
lstm_model_file = 'lstm_cnn_model.h5' 

# Data Generator
imgSeqGenObj = ImageSeqGenerator()


def train_cnn(cnn_model_file):
	# import Data
	X_train, y_train = imgSeqGenObj.mnist_train_data()
	X_test, y_test = imgSeqGenObj.mnist_test_data()

	# build the model
	cnn_model = create_model.cnn(model_file=None, output_layer=True, trainable=True)

	# Compile model
	cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=100, verbose=2)

	# Final evaluation of the model
	scores = cnn_model.evaluate(X_test, y_test, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))

	cnn_model.save_weights(cnn_model_file)

	return cnn_model

def train_lstm(cnn_model_file, lstm_model_file,seq_len=3):

	# Check if Pre-trained CNN exists
	if not os.path.exists(cnn_model_file):
		print("Pre-Trained CNN not found: Please train CNN first.")
		sys.exit()

	# Import data
	X_seq_train, y_seq_train = imgSeqGenObj.gen_train_sequences(seq_len=seq_len,n_samples=10000)
	X_seq_test, y_seq_test = imgSeqGenObj.gen_test_sequences(seq_len=seq_len,n_samples=1000)

	# Build Model
	lstm_cnn_model = create_model.lstm_cnn( model_file=None, MAX_SEQ_LEN=None, cnn_model_file=cnn_model_file)

	# Compile model
	lstm_cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Fit the model
	lstm_cnn_model.fit(X_seq_train, y_seq_train, validation_data=(X_seq_test, y_seq_test), epochs=5, batch_size=100, verbose=2)

	lstm_cnn_model.save_weights(lstm_model_file)

	return lstm_cnn_model

def test_cnn(cnn_model_file):
	if not os.path.exists(cnn_model_file):
		print("Model file Note found!!")
		sys.exit()
	# import Data
	X_test, y_test = imgSeqGenObj.mnist_test_data()

	# build the model
	cnn_model = create_model.cnn(model_file=cnn_model_file, output_layer=True, trainable=True)

	# Compile model
	cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# Final evaluation of the model
	scores = cnn_model.evaluate(X_test, y_test, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))

	return

def test_lstm(lstm_model_file, seq_len=3):
	if not os.path.exists(lstm_model_file):
		print("Model file Note found!!")
		exit()

	seq_len = max(2,seq_len)
	# build the model
	lstm_model = create_model.lstm_cnn(model_file=lstm_model_file, MAX_SEQ_LEN=None, cnn_model_file=None)

	lstm_model.summary()
	# import data
	X_seq_test, y_seq_test = imgSeqGenObj.gen_test_sequences(seq_len=seq_len,n_samples=1)

	# inference
	output = lstm_model.predict(X_seq_test)

	gt_label = y_seq_test[0,:].argmax()
	out_label = output[0,:].argmax()
	fig = plt.figure(1)
	for i in range(seq_len):
		plt.subplot(1,seq_len,i+1)
		plt.imshow(X_seq_test[0,i,0,:,:],cmap='gray')

	conf = output[0,out_label]
	if gt_label == out_label:
		fig.suptitle('Correct:  Pred/GT = {}/{}, Conf: {:01.3f}'.format(out_label, gt_label, conf), fontsize = 12)
	else:
		fig.suptitle('Incorrect: Pred/GT = {}/{}, Conf: {:01.3f}'.format(out_label, gt_label, conf), fontsize = 12)
	plt.show()


	return


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Script to Train and Test a sample CNN+LSTM model in Keras")
	parser.add_argument("mode", help="training or testing mode")
	parser.add_argument("model", help="Model type CNN or LSTM")
	parser.add_argument("-l","--length",default=3,help="Length of image sequence")

	args = parser.parse_args()

	seq_len = int(args.length)
	if args.mode.lower() == 'train':
		if args.model.lower() == 'cnn':
			train_cnn(cnn_model_file)
		elif args.model.lower() == 'lstm':
			train_lstm(cnn_model_file, lstm_model_file,seq_len=seq_len)
		else:
			print('Unknown Model type...')
	elif args.mode.lower() == 'test':
		if args.model.lower() == 'cnn':
			test_cnn(cnn_model_file)
		elif args.model.lower() == 'lstm':
			test_lstm(lstm_model_file, seq_len=seq_len)
		else:
			print('Unknown Model type...')	
	else:
		print('Unknown Mode ...')	