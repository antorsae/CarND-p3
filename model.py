from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.noise import GaussianNoise
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D, AveragePooling2D
from keras.layers.advanced_activations import ELU

from keras.activations import relu

from keras.optimizers import Adam
import json
from keras.models import model_from_json

from preprocess import Preprocess
from keras import backend as K

def max_abs_error(y_true, y_pred):
    return K.max(K.abs(y_pred - y_true))

def get_model(which='nvidia'):

	sizex, sizey = (Preprocess.sizex, Preprocess.sizey)

	if which == 'nvidia':	
		model = get_model_nvidia(sizex, sizey)
	elif which == 'comma':
		model = get_model_comma(sizex, sizey)
	
	model.compile(
		loss='mse', 
		metrics=['mse', max_abs_error],
	optimizer=Adam(lr=1e-4))

	model.summary()

	return model

def get_model_nvidia(sizex, sizey):
	model = Sequential()

	model.add(Cropping2D(cropping=((56, 24), (0, 0)),input_shape=(sizey, sizex, 3)))
	model.add(AveragePooling2D(pool_size=(1, 2)))
	model.add(Lambda(minmax_norm))
#	model.add(GaussianNoise(0.2))

	init = 'glorot_normal'
	activation = 'relu'

	model.add(Convolution2D(24, 10, 10, subsample = (2,2), border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(16, 10, 10, subsample = (2,2), border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(16, 5, 5, subsample = (2,2), border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(16, 3, 3, border_mode='valid', init = init))
	model.add(Activation(activation))

	model.add(Convolution2D(16, 3, 3, border_mode='valid', init = init))
	model.add(Activation(activation))

	#model.add(Convolution2D(64, 3, 3, border_mode='valid', init = init))
	#model.add(Activation(activation))

	model.add(Flatten())

	model.add(Dropout(.2))
	model.add(Dense(64, init = init))
	model.add(Activation(activation))

	model.add(Dropout(.3))
	model.add(Dense(32, init = init))
	model.add(Activation(activation))

	model.add(Dropout(.4))
	model.add(Dense(16, init = init))
	model.add(Activation(activation))

	model.add(Dropout(.5))
	model.add(Dense(8, init = init))
	model.add(Activation(activation))

	model.add(Dense(1))

	return model

def minmax_norm(x):
    xmin = K.min(x, axis=[1,2,3], keepdims=True)
    xmax = K.max(x, axis=[1,2,3], keepdims=True)
    
    return (x - xmin ) / (xmax-xmin) - 0.5

def get_model_comma(sizex, sizey):
	activation = 'elu'
	init = 'glorot_uniform'

	model = Sequential()
	model.add(Cropping2D(cropping=((56, 24), (0, 0)),input_shape=(sizey, sizex, 3)))
	model.add(Lambda(minmax_norm))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", init = init))
	model.add(Activation(activation))
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", init = init))
	model.add(Activation(activation))
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", init = init))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(Activation(activation))
	model.add(Dense(512, init = init))
	model.add(Dropout(.5))
	model.add(Activation(activation))
	model.add(Dense(1))
	
	return model
