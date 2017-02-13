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
import random

def balance(log):
	balanced = pd.DataFrame()   # Balanced dataset
	bins = 1000                 # N of bins
	bin_n = 300                 # N of examples to include in each bin (at most)

	start = 0
	for end in np.linspace(0, 1, num=bins):  
		df_range = log[(np.absolute(log.steering) >= start) & (np.absolute(log.steering) < end)]
		range_n = min(bin_n, df_range.shape[0])
		if range_n > 0:
			balanced = pd.concat([balanced, df_range.sample(range_n)])
		start = end
	return balanced

def generator(log, validation = False):
	
	shown = 0
	
	while True:
		log = shuffle(log)

		if validation == False:
			print("Generator loop") # print to make sure we start yielding accross each epoch (debug only)
			balanced = log #balance(log)
		else:
			balanced = log
		hot = 0

		
		for i, ll in balanced.groupby(np.arange(len(balanced)) // BS):
				
			#images              = np.empty([0, Preprocess.sizey, Preprocess.sizex, 3], dtype=np.uint8)
			images              = np.empty([0, 160, 320, 3], dtype=np.uint8)
			augmented_steerings = np.empty([0, 1], dtype=np.float32)

			for j,l in ll.iterrows():
				center = imread(l.center)
				st = l.steering

				images 				= np.vstack((images, [center]))
				augmented_steerings = np.vstack((augmented_steerings, [st]))

				if validation == False:
					total_transforms = l.augment_transforms.split()
				#	for t in range(11-len(total_transforms)):
				#		total_transforms.append(total_transforms[t % len(total_transforms)])
					for augment_transform in total_transforms:
						flip = False
						if augment_transform == 'cf':
							st = -l.steering
							images 				= np.vstack((images, [cv2.flip(center, flipCode = 1)]))
							augmented_steerings = np.vstack((augmented_steerings, [st]))
						elif (augment_transform.startswith('cs')):
							px = (15.) * np.random.random_sample() + 7.5
							st = l.steering - px * 0.004
							if augment_transform.endswith('f'):
								st = -st
								flip = True
							shifted_image       = cv2.warpAffine(center, np.float32([[1,0,px],[0,1,0]]), (center.shape[1],center.shape[0]))
							images 				= np.vstack((images, [shifted_image if not flip else cv2.flip(shifted_image, flipCode = 1)]))
							augmented_steerings = np.vstack((augmented_steerings, [st]))
						elif augment_transform.startswith('r'):
							st = l.steering + l.steering_bias_right
							if augment_transform.endswith('f'):
								st = -st
								flip = True
							right = imread(l.right)
							images 				= np.vstack((images, [right if not flip else cv2.flip(right, flipCode = 1)]))
							augmented_steerings = np.vstack((augmented_steerings, [st]))

						elif augment_transform.startswith('l'):
							st = l.steering + l.steering_bias_left
							if augment_transform.endswith('f'):
								st = -st
								flip = True
							left = imread(l.left)			
							images 				= np.vstack((images, [left if not flip else cv2.flip(left, flipCode = 1)]))
							augmented_steerings = np.vstack((augmented_steerings, [st]))
						elif (augment_transform	.startswith('rs')):
							px = np.abs((15.) * np.random.random_sample())
							st = l.steering +steering_bias_right - px * 0.004
							if augment_transform.endswith('f'):
								st = -st
								flip = True
							right = imread(l.right)
							shifted_image       = cv2.warpAffine(right, np.float32([[1,0,px],[0,1,0]]), (right.shape[1],right.shape[0]))
							images 				= np.vstack((images, [shifted_image if not flip else cv2.flip(shifted_image, flipCode = 1)]))
							augmented_steerings = np.vstack((augmented_steerings, [st]))
						elif (augment_transform	.startswith('ls')):
							px = -np.abs((15.) * np.random.random_sample())
							st = l.steering +steering_bias_left - px * 0.004
							if augment_transform.endswith('f'):
								st = -st
								flip = True
							left = imread(l.left)
							shifted_image       = cv2.warpAffine(left, np.float32([[1,0,px],[0,1,0]]), (left.shape[1],left.shape[0]))
							images 				= np.vstack((images, [shifted_image if not flip else cv2.flip(shifted_image, flipCode = 1)]))
							augmented_steerings = np.vstack((augmented_steerings, [st]))
					
			
			for (image, augmented_steering) in zip(images, augmented_steerings):
				h, w = image.shape[0], image.shape[1]
				[x1, x2] = np.random.choice(w, 2, replace=False)
				k = h / (x2 - x1)
				b = - k * x1
				shadow = (np.random.random_sample() * 0.2 + 0.3)
				side = random.choice([True, False])
				shadowed = np.empty_like(image)
				for hi in range(h):
					c = int((hi - b) / k)
					if side:
						shadowed[hi, c:, :] = (image[hi, c:, :] * shadow).astype(np.int32)
					else:
						shadowed[hi, :c, :] = (image[hi, :c, :] * shadow).astype(np.int32)
				images 				= np.vstack((images, [shadowed]))
				augmented_steerings = np.vstack((augmented_steerings, [augmented_steering]))

			images_processed = np.empty([0, Preprocess.sizey, Preprocess.sizex, 3], dtype=np.uint8)

			for im in images:
				images_processed = np.vstack((images_processed, [Preprocess.preprocess(im)]))
			#	for num, image in enumerate(images):
		#			imsave( "trans-{}.jpg".format(num), image)


			yield (images_processed, np.clip(augmented_steerings, -1., 1.))

# ***** main loop *****

if __name__ == "__main__":

	BS = 2

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

	#imread('driving-right/'+driving_log.center.sample().values[0])

#	def keep_only(log, keep):
#		return pd.concat([log[log.steering==0].sample(frac=keep),log[log.steering!=0]])
	
	# get rid of 95% of zero center steering
	#center_log = keep_only(center_log, 0.40)
	#left_log   = keep_only(left_log, 0.15)
	#right_log  = keep_only(right_log, 0.55)

	steering_bias = 0.2

	# cf  = center flipped
	# l   = left camera
	# r   = right camera
	# lf  = left camera flipped
	# rf  = right camera flipped
	# cs  = center shift
	# csf = center shift flipped 

	# driving in the center, steering follows the road
	center_log['steering_bias_left']  =  steering_bias
	center_log['steering_bias_right'] = -steering_bias
	center_log['augment_transforms'] = "cf l r lf rf" #cs csf ls lsf rs rsf" #8

	# driving at the left edge of the road, steering follows the road
	left_log.steering  = left_log.steering + 0.8
	left_log['steering_bias_left']  =       steering_bias
	left_log['steering_bias_right'] = -1. * steering_bias
	left_log['augment_transforms'] = "cf r rf"# rs rsf cs csf" #4

	# driving at the right edge of the road, steering follows the road
	right_log.steering = right_log.steering - 0.8
	right_log['steering_bias_left']  =       steering_bias
	right_log['steering_bias_right'] =  -1. * steering_bias
	right_log['augment_transforms'] = "cf l lf"# ls lsf cs csf" #4
	
	# fragments of attempting to drive out of the road, pointing ~15 deg to the right
	# steering is 0
	sk_right_log.steering = -0.6
	sk_right_log['steering_bias_left'] =        steering_bias
	sk_right_log['steering_bias_right'] = -1. * steering_bias
	sk_right_log['augment_transforms'] = "cf r rf l lf"# rs rsf" #4

	# fragments of attempting to drive out of the road, pointing ~15 deg to the left
	# steering is 0
	sk_left_log.steering = 0.6
	sk_left_log['steering_bias_left'] =    steering_bias
	sk_left_log['steering_bias_right'] = -steering_bias
	sk_left_log['augment_transforms'] = "cf l lf r rf"# ls lsf" #4

	driving_log = balance(pd.concat([center_log, left_log, right_log, sk_right_log, sk_left_log]))
	print(np.histogram(np.abs(driving_log.steering), bins=100))

	print((driving_log.augment_transforms.str.split().str.len()+1).sum())

	train_log, validate_log = train_test_split(driving_log, test_size=0.10)
	print("Samples in train set:", train_log.shape[0])

	cb   = ModelCheckpoint(current_model + ".h5", monitor='val_loss', save_best_only=False)
	cbw  = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))

	model = get_model(current_model)

	with open(current_model + ".json", "w") as file:
		file.write(model.to_json())

	history = model.fit_generator(generator(train_log), 
				nb_epoch=100, 
				samples_per_epoch = 2*(train_log.augment_transforms.str.split().str.len()+1).sum(),
				validation_data = generator(validate_log, validation=True),
				nb_val_samples = validate_log.shape[0],
				callbacks = [cb])

