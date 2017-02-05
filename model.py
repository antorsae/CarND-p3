from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.activations import relu, elu
from keras.optimizers import Adam

from preprocess import size

sizex, sizey = size()

def get_model():
	model = get_model_comma()

	model.compile(
	    loss='mean_squared_error', 
	    metrics=['mean_squared_error'], 
	    optimizer=Adam(lr=0.001))

	return model

def get_model_nvidia():
	model = Sequential()

	model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(sizey, sizex, 3)))

	init = 'glorot_normal'
	activation = 'elu'

	model.add(Convolution2D(24, 5, 5, subsample = (2,2), border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(36, 5, 5, subsample = (2,2), border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(48, 5, 5, subsample = (2,2), border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(64, 3, 3, border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(64, 3, 3, border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Flatten())

	model.add(Dense(1164, init = init))
	model.add(Activation(activation))

	model.add(Dense(100, init = init))
	model.add(Activation(activation))

	model.add(Dense(50, init = init))
	model.add(Activation(activation))

	model.add(Dense(10, init = init))
	model.add(Activation(activation))

	model.add(Dense(1))

	return model

def get_model_comma():
	activation = 'elu'

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
	            input_shape=(sizey, sizex, 3)))

	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(Activation(activation))
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation(activation))
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(Activation(activation))
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(Activation(activation))
	model.add(Dense(1))
	
	return model
