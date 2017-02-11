import time
import tensorflow as tf
tf.python.control_flow_ops = tf

import numpy as np
import pandas as pd
import preprocess
from preprocess import Preprocess

import cv2
from sklearn.utils import shuffle
from scipy.misc import imread, imsave

from model import get_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.callbacks import LambdaCallback
import argparse

def generator(log, validation = False, steering_corr_bias   = 0.20, steering_corr_weight = 0.0):
	
	shown = 0
	
	while True:
		log = shuffle(log)
		if validation == False:
			print("LOOP")
		
		for i, ll in log.groupby(np.arange(len(log)) // BS):
				
			images              = np.empty([0, Preprocess.sizey, Preprocess.sizex, 3], dtype=np.uint8)
			augmented_steerings = np.empty([0, 1], dtype=np.float32)

			for j,l in ll.iterrows():
				center = imread(l.center)
				images 				= np.vstack((images, [Preprocess.preprocess(center)]))
				augmented_steerings = np.vstack((augmented_steerings, [l.steering]))

				if validation == False:
					for augment_transform in l.augment_transforms.split():
						if augment_transform == 'cf':
							images 				= np.vstack((images, [Preprocess.preprocess(cv2.flip(center, flipCode = 1))]))
							augmented_steerings = np.vstack((augmented_steerings, [-1. * l.steering]))
						elif augment_transform == 'cs':
							px = (60.) * np.random.random_sample() + 30.
							shifted_image       = cv2.warpAffine(center, np.float32([[1,0,px],[0,1,0]]), (center.shape[1],center.shape[0]))
							#print(shifted_image.shape)
							images 				= np.vstack((images, [Preprocess.preprocess(shifted_image)]))
							augmented_steerings = np.vstack((augmented_steerings, [l.steering - px * 0.004]))
						elif augment_transform == 'r':
							images 				= np.vstack((images, [Preprocess.preprocess(imread(l.right))]))
							augmented_steerings = np.vstack((augmented_steerings, [l.steering + l.steering_bias_right]))
						elif augment_transform == 'l':
							images 				= np.vstack((images, [Preprocess.preprocess(imread(l.left))]))
							augmented_steerings = np.vstack((augmented_steerings, [l.steering + l.steering_bias_left]))

			yield (images, np.clip(augmented_steerings, -1., 1.))


# ***** main loop *****

if __name__ == "__main__":

	BS = 8

	parser = argparse.ArgumentParser(description='Train behavioral cloning udacity CarND P3')
	parser.add_argument('centerdir', type=str, default='driving-centered', help='Directory name of training data for CENTERED driving')
	parser.add_argument('leftdir', type=str, default='driving-left', help='Directory name of training data for driving on the LEFT')
	parser.add_argument('rightdir', type=str, default='driving-right', help='Directory name of training data for driving on the RIGHT')
	parser.add_argument('model', type=str, default="comma", help='Model (nvidia, comma)')
	args = parser.parse_args()

	current_model = args.model
	print('Using model: ',current_model)

	# Train the model
	# History is a record of training loss and metrics
	center_log = pd.read_csv(args.centerdir+'/driving_log.csv')
	left_log   = pd.read_csv(args.leftdir+'/driving_log.csv')
	right_log  = pd.read_csv(args.rightdir+'/driving_log.csv')
	
	sk_right_dir = 'driving-skewed-right-15'
	sk_right_log  = pd.read_csv(sk_right_dir+'/driving_log.csv')

	sk_left_dir = 'driving-skewed-left-15'
	sk_left_log  = pd.read_csv(sk_left_dir+'/driving_log.csv')
	
	for i in ['left', 'right', 'center']:
		center_log[i]   = args.centerdir + '/' + center_log[i].str.strip()
		left_log[i]     = args.leftdir   + '/' + left_log[i].str.strip()
		right_log[i]    = args.rightdir  + '/' + right_log[i].str.strip()
		sk_right_log[i] = sk_right_dir   + '/' + sk_right_log[i].str.strip()
		sk_left_log[i]  = sk_left_dir    + '/' + sk_left_log[i].str.strip()

	def keep_only(log, keep):
		return pd.concat([log[log.steering==0].sample(frac=keep),log[log.steering!=0]])
	
	# get rid of 95% of zero center steering
	#center_log = keep_only(center_log, 0.40)
	#left_log   = keep_only(left_log, 0.15)
	#right_log  = keep_only(right_log, 0.55)

	steering_bias = 0.22

	center_log['steering_bias_left']  =  steering_bias
	center_log['steering_bias_right'] = -steering_bias
	center_log['augment_transforms'] = "cf l r cs"

	left_log.steering  = left_log.steering + 0.7
	left_log['steering_bias_left']  =       steering_bias
	left_log['steering_bias_right'] = -1. * steering_bias
	left_log['augment_transforms'] = "cf r"

	right_log.steering = right_log.steering - 0.7
	right_log['steering_bias_left']  =       steering_bias
	right_log['steering_bias_right'] =  -1. * steering_bias
	right_log['augment_transforms'] = "cf l"
	
	sk_right_log['steering'] = -0.5
	sk_right_log['steering_bias_left'] =        steering_bias
	sk_right_log['steering_bias_right'] = -1. * steering_bias
	sk_right_log['augment_transforms'] = "cf r"

	sk_left_log['steering'] = 0.5
	sk_left_log['steering_bias_left'] = steering_bias
	sk_left_log['steering_bias_right'] = -steering_bias
	sk_left_log['augment_transforms'] = "cf l"

	driving_log = pd.concat([center_log, left_log, right_log, sk_right_log, sk_left_log])

	print((driving_log.augment_transforms.str.split().str.len()+1).sum())

	train_log, validate_log = train_test_split(driving_log, test_size=0.10)
	print("Samples in train set:", train_log.shape[0])

	cb   = ModelCheckpoint(current_model + ".h5", monitor='val_loss', save_best_only=False)
	cbw  = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))

	model = get_model(current_model)

	with open(current_model + ".json", "w") as file:
		file.write(model.to_json())

	#augment_transforms = ['cf', 'l', 'r']

	history = model.fit_generator(generator(train_log), 
				nb_epoch=40, 
				samples_per_epoch = (train_log.augment_transforms.str.split().str.len()+1).sum(),
				validation_data = generator(validate_log, validation=True),
				nb_val_samples = validate_log.shape[0],
				callbacks = [cb])

