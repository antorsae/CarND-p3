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
from keras.callbacks import LambdaCallback, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import argparse
import random

BINS = 	100
MAX_PER_BIN = 300
SMOOTHING = True

# given a log entry generates 12 different transformations:
# - image, blurred image (x3), blurred image (x5)
# - shadowed and flipped for all of the 3 above
def gen_transforms(l):
	st        = l.steering
	image     = imread(l.image)
	image_f   = cv2.flip(image, flipCode = 1) 
	image_b3  = cv2.GaussianBlur(image,   (3,3),0 ) 
	image_b3f = cv2.GaussianBlur(image_f, (3,3),0 )
	image_b5  = cv2.GaussianBlur(image,   (5,5),0 ) 
	image_b5f = cv2.GaussianBlur(image_f, (5,5),0 ) 

	return (image, shadow(image), image_f, shadow(image_f), image_b3, shadow(image_b3), image_b3f, shadow(image_b3f), image_b5, shadow(image_b5), image_b5f, shadow(image_b5f),
			   st,  		  st,     -st,			   -st,   	  st, 	            st,	      -st,	             -st,       st,			      st,       -st,              -st)

# GENERATOR
# 
# yield images and steering angles for training and validation
# takes log with images and steerings, and will augment data
# if using in training.
def generator(log, log_left=None, validation = False):
	
	save_counter = 0
	
	while True:
		log = shuffle(log)

		for i, ll in log.groupby(np.arange(len(log)) // BS):

			# incoming size of images is 160x320 (hardcoded!)
			images              = np.empty([0, 160, 320, 3], dtype=np.uint8)
			augmented_steerings = np.empty([0, 1], dtype=np.float32)

			work_l = []
			work_a = []
			for j,l in ll.iterrows():
				if validation == True:
					image = imread(l.image)
					st = l.steering
					images 				= np.vstack((images, [image]))
					augmented_steerings = np.vstack((augmented_steerings, [st]))
				else:
					work_l.extend([l])

			if (validation == False):

				# nvidia-smi reports low GPU usage (<50%) so I tried increasing
				# batch size... but as I increased batch size EPOCH times grew
				# this led me to think we were CPU-bound, so I experimented 
				# with multi-threading.
				# 
				# Experiment was a failure. leave to False since MT is significantly 
				# slower than ST still don't know WHY. I also tried this generator in 
				# multiprocess mode (see Keras dox: nb_worker, pickle_safe) but results 
				# in slower behavior too.
				multi_thread = False

				if multi_thread:
					from multiprocessing.dummy import Pool as ThreadPool 
					from multiprocessing import cpu_count

					pool = ThreadPool(cpu_count())
					results = pool.starmap(gen_transforms, work_l)
					pool.close()
					pool.join()
				else:
					results = []
					for w_l in work_l:
						results.append(gen_transforms(w_l))
				for r in results:
					# todo: infer augment factor autmatically
					images 				= np.vstack((images, 			  [r[ 0]], [r[ 1]], [r[ 2]], [r[ 3]], [r[ 4]], [r[ 5]], [r[ 6]], [r[ 7]], [r[ 8]], [r[ 9]], [r[10]], [r[11]]))
					augmented_steerings = np.vstack((augmented_steerings, [r[12]], [r[13]], [r[14]], [r[15]], [r[16]], [r[17]], [r[18]], [r[19]], [r[20]], [r[21]], [r[22]], [r[23]]))

			images_processed = np.empty([0, Preprocess.sizey, Preprocess.sizex, 3], dtype=np.uint8)

			# sporadically save augmented images for manual inspection
			for inum,im in enumerate(images):
				if (save_counter % 1000) == 0:
					imsave( "train-{}.jpg".format(inum), im)
				images_processed = np.vstack((images_processed, [Preprocess.preprocess(im)]))

			save_counter += 1

			# shuffle so we are not always yielding augmented images in same order
			(images_processed, augmented_steerings) = shuffle(images_processed, augmented_steerings)

			yield (images_processed, np.clip(augmented_steerings, -1., 1.))

		if validation == False:
			log, log_left = balance(log, log_left)
			print() 
			print("Balanced items:", len(log), "Unbalanced items left for next epochs:", len(log_left), "(for sequential rebalancing)")


# balances dataset by undersampling: take as much as bin_n items for 
# each bin. inspired by: http://navoshta.com/end-to-end-deep-learning/
# Added support for sequential balancing: returns balanced set as 
# well as items left from unbalanced set for later re-balancing
# http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tsmcb09.pdf
#
# pros: all items are used after N re-balances
# todo: all items are just seen once, it would be better to make sure
#       items are sequentially seen later as often as possible. 
def balance(b, ul = None):
	balanced = pd.DataFrame()   # Balanced dataset
	bins =  BINS                # N of bins
	bin_n = MAX_PER_BIN         # N of examples to include in each bin (at most)
	if ul is None:
		ul = pd.DataFrame(b) #data=None, columns=unbalanced.columns)
		b  = pd.DataFrame(data=None, columns=b.columns)
	start = 0
	for end in np.linspace(0, 1, num=bins):  
		if SMOOTHING:
			bin_n = int( MAX_PER_BIN * np.clip(-2 * start + 1.9, 0.1,1.))
		ul_  = ul[(np.absolute(ul.steering) >= start) & (np.absolute(ul.steering) < end)]
		ul_n = min(bin_n, ul_.shape[0])
		b_   =  b[(np.absolute(b.steering)  >= start) &  (np.absolute(b.steering) < end)]
		b_n  = min(bin_n,  b_.shape[0])
		if ul_n > 0:
			ul_  = ul_.sample(ul_n)
			balanced = pd.concat([balanced, ul_, b_[:(bin_n-ul_n)]])
			ul = ul[~ul['image'].isin(ul_['image'])]
		else:
			balanced = pd.concat([balanced, b_])
		start = end
	return balanced, ul

# apply random shadow to image (andom vertex and intensity)
# loosely based on: http://navoshta.com/end-to-end-deep-learning/
def shadow(image):
	max_shadow_sides = np.random.randint(2,10)
	h, w = image.shape[0], image.shape[1]
	y = np.append(np.unique(np.random.randint(1, h, size=max_shadow_sides)),h)
	shadow_sides=len(y)
	y[1:] = y[1:]-y[0:-1]
	x = np.random.choice(w, shadow_sides+1, replace=False)
	hii = 0
	shadow_image = np.array(image)
	side = random.choice([True, False])
	shadow = (np.random.random_sample() * 0.2 + 0.3)
	for n in range(shadow_sides):
		k = y[n] / (x[n+1] - x[n])
		b = - k * x[n]
		for hi in range(y[n]):
			c = int((hi - b) / k)
			if side:
				shadow_image[hii, c:, :] = (image[hii, c:, :] * shadow).astype(np.uint8)
			else:
				shadow_image[hii, :c, :] = (image[hii, :c, :] * shadow).astype(np.uint8)
			hii += 1
	return shadow_image

# takes CSV and outouts DF with 'image' and 'steering'
# adjust steering based on 'cb' (center bias) and lrb (left/right bias) 
# if applicable
def read_log(dir, t="c", cb=0., lrb=0.1):
	log = pd.read_csv(dir+'/driving_log.csv')
	new = pd.DataFrame()
	m = { "c" : "center", "l" : "left", "r" : "right"}
	for i in t.split():
		camera = log[[m[i], 'steering']].copy().rename(columns={m[i]: 'image'})
		if i == 'c':
			bias =  cb
		elif i == 'l':
			bias =  cb + lrb
		elif i == 'r':
			bias =  cb - lrb
		camera.steering += bias
		camera.image     = dir + '/' + camera.image.str.strip()
		new = pd.concat([new, camera])
	return new

# ***** main loop *****

if __name__ == "__main__":

	BS = 1

	parser = argparse.ArgumentParser(description='Train behavioral cloning udacity CarND P3')
	parser.add_argument('centerdir', type=str, default='driving-centered', help='Directory name of training data for CENTERED driving')
	parser.add_argument('leftdir', type=str, default='driving-left', help='Directory name of training data for driving on the LEFT')
	parser.add_argument('rightdir', type=str, default='driving-right', help='Directory name of training data for driving on the RIGHT')
	parser.add_argument('model', type=str, default="comma", help='Model (nvidia, comma)')
	args = parser.parse_args()

	# todo: argparse this
	sk_right_dir = 'driving-skewed-right-15'
	sk_left_dir  = 'driving-skewed-left-15'
	track2_dir   = 'track2-validation'

	current_model = args.model
	print('Using model: ',current_model)

	center_log   = read_log(args.centerdir, t = "c l r")

	left_log     = read_log(args.leftdir,   t = "c r"  , cb =  0.5)
	right_log    = read_log(args.rightdir,  t = "c l",   cb = -0.5)

	sk_left_log  = read_log(sk_left_dir,    t = "c l r", cb =  0.6)
	sk_right_log = read_log(sk_right_dir,   t = "c l r", cb = -0.6)

	track2_log   = read_log(track2_dir,     t = "c l r")

	cb   = ModelCheckpoint(current_model + ".h5", monitor='val_loss', save_best_only=True)
	cbtb = TensorBoard(write_images=True)

	model = get_model(current_model)

	with open(current_model + ".json", "w") as file:
		file.write(model.to_json())

	driving_log = shuffle(pd.concat([center_log, left_log, right_log, sk_right_log, sk_left_log]))
	train_log, validate_log = train_test_split(driving_log, test_size=0.20)

	print("Samples in unbalanced train set:",      train_log.shape[0])
	print("Samples in unbalanced validation set:", validate_log.shape[0]) #only track #1

	b_train_log, b_train_log_left    = balance(train_log)

	import matplotlib
	matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
	import matplotlib.pyplot as plt
	plt.hist(np.abs(train_log['steering']), bins=BINS-1, range=(0.,1.))
	plt.savefig("train-unbalanced.png")
	plt.gcf().clear()

	plt.hist(np.abs(b_train_log['steering']), bins=BINS-1, range=(0.,1.))
	plt.savefig("train-balanced.png")
	
	b_validate_log = pd.concat([balance(validate_log)[0], balance(track2_log)[0]])
	print("Samples in balanced train set:",      b_train_log.shape[0])
	print("Samples in balanced validation set:", b_validate_log.shape[0])
	history = model.fit_generator(generator(b_train_log, b_train_log_left), 
				nb_epoch=200, 
				samples_per_epoch = 12*b_train_log.shape[0],
				validation_data = generator(b_validate_log, validation=True),
				nb_val_samples = b_validate_log.shape[0],
				callbacks = [cb, cbtb])

