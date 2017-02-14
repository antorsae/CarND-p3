import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

import datetime as dt

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

from preprocess import Preprocess

n1 = dt.datetime.now()
n2 = dt.datetime.now()

@sio.on('telemetry')
def telemetry(sid, data):
	global n1,n2
	# The current steering angle of the car
	steering_angle = data["steering_angle"]
	# The current throttle of the car
	throttle = float(data["throttle"])
	# The current speed of the car
	speed = float(data["speed"])
	# The current image from the center camera of the car
	imgString = data["image"]
	image = Image.open(BytesIO(base64.b64decode(imgString)))
	image_array = Preprocess.preprocess(np.asarray(image))
	transformed_image_array = image_array[None, :, :, :]
	# This model currently assumes that the features of the model are just the images. Feel free to change this.

	n1=dt.datetime.now()
	telemetry_time = (n1 - n2).microseconds / 1000

	steering_angle = float(model.predict(transformed_image_array, batch_size=1))
	n2=dt.datetime.now()
	prediction_time = (n2 - n1).microseconds / 1000

	#time.sleep(1)
	# The driving model currently just outputs a constant throttle. Feel free to edit this.

	# speed_up_angle is 0 when angle is too abrupt to speed up and 1 when it's ok (heading straight)
	speed_up_angle = np.max([0.15 - np.abs(steering_angle), 0.]) / 0.15

	if speed < 10.0:
		throttle += 0.1
	else:
		throttle = 0.25 * speed_up_angle + 0.01

	print("Angle: ",steering_angle, " Throttle: ", throttle, "Speed: ", speed, " Pred time: ", prediction_time, " Tele time: ", telemetry_time)

	send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
	print("connect ", sid)
	send_control(0, 0)


def send_control(steering_angle, throttle):
	sio.emit("steer", data={
	'steering_angle': steering_angle.__str__(),
	'throttle': throttle.__str__()
	}, skip_sid=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Remote Driving')
	parser.add_argument('model', type=str,
	help='Path to model definition json. Model weights should be on the same path.')
	args = parser.parse_args()
	with open(args.model, 'r') as jfile:
		# NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
		# then you will have to call:
		#
		#   model = model_from_json(json.loads(jfile.read()))\
		#
		# instead.
		model = model_from_json(jfile.read())


	model.compile("adam", "mse")
	weights_file = args.model.replace('json', 'h5')
	model.load_weights(weights_file)

	# wrap Flask application with engineio's middleware
	app = socketio.Middleware(sio, app)

	# deploy as an eventlet WSGI server
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)