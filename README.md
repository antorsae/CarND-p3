##Behavioral Cloning Project##

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Overview and project deliverables

My project includes the following files:
* model.py contains two models: one based on NVIDIA and another model based on COMMA.AI. I wanted to test how different models behave using the same training data.
* drive.py for driving the car in autonomous mode. It accepts a command line parameter for which model to load as well as an optional parameter to set the target speed (otherwise it detaults to 20 mph), e.g.

```sh
python drive.py nvidia.json 25
```

Once it connects to the simulator in autonomous mode, it outputs the predicted steering angle, speed, as well as prediction time (in milliseconds),  the time since the last frame -tele time-, as well as the net prediction fps:
![drive.py output](./assets/driving.png)
* preprocess.py just includes common preprocessing code in Python but at the end I did included all preprocessing in the model itself. I left it in for future use.
* Trained models:
- comma.json/h5 contains the comma.ai model trained using centered, left-sided, right-sided and skewed driving (more on this later).
- nvidia.json/h5 contains the nvidia-inspired trained using centered, left-sided, right-sided and skewed driving.
- nvidia-centered.json/h5 same model trained only with centered driving.
- nvidia-centered-left-right.json/h5 same model trained with centered driving and left-sided and right-sided driving (but not skewed driving)
- nvidia-left-right.json.json/h5 same model trained only with left-sided and right-sided driving
- nvidia-left-right-skleft-skright.json.json/h5 same model trained with left-sided, right-sided and skewed driving
- readme.md (this file)

### Results

Click on the GIFs to see full-version on youtube.

[![Track #1](./assets/track1-30mph-fantastic.gif "Track #1")](https://www.youtube.com/watch?v=vDBoPAgClQo) [![Track #2](./assets/track2-30mph-fast.gif "Track #1")](https://www.youtube.com/watch?v=)

#### 1. Training data
This project involved collecting good driving behavior from the Udacity simulator to train a neural network to learn to predict the steering angle. The simulator collects both the center image as seen inside car as well as left and right images.

#### Learning from good driving behavior
I collected "good" driving behavior driving in the center of the track. I collected a total of ~27k datapoints driving as best as I could.

Here's how the centered driving looks like (left, center and right cameras):

![Centered driving](./assets/centered-LCR.gif "Centered Driving")

The train.py contains code to load the above training data and use the center, left and right cameras:

```
center_log   = read_log(args.centerdir, t = "c l r")
```

The read_log function will load the data and use all three cameras as indicated by "c l r". The steering angle is left as-is for the center camera and is corrected by configurable bias:

```
# takes CSV and outouts DF with 'image' and 'steering'
# adjust steering based on 'cb' (center bias) and lrb (left/right bias) 
# if applicable
def read_log(dir, t="c", cb=0., lrb=0.18):
```

In the code above the left / right bias defaults to 0.18. This bias controls how fast we want the car to get to a point ahead of the center camera. If the number is too low, the car will steer back to the center slowly; and if it's too high it will go back more quickly. 

The performance of the system where you run the simulator and the drive.py script severely impacts actual driving. If the computer is slow, the correction bias is high (e.g. 0.15 or more), and the speed is high (e.g. 20 mph) the car will end up driving in curves. This is because the steering angle will only be predicted and adjusted a few times per second so a really long straight line will be needed to stabilize the car. The drive.py outputs effective prediction fps to help diagnose performance issues:

![drive.py output fps](./assets/driving-fps-detail.png)

Moreover, the steering angles output to the simulator are repeated either 2x or 3x. I believe there this is a bug in the simulator or drive.py supplied code and the frame gathered by the telemetry function is not updated on every call, so the net predictions per seconds are half or a third of those returned. In the output above, the reported 12 fps should be divided by 3: the steering prediction is repeated each time so the net speed at which the steering angle is adjusted is just 4 times per second! 

#### Augmented behavior: Learning from bad behavior

Although there's already implicit corrections made by using the left and right cameras, good driving is not enough: should a small error occur, the network will not know how to steer back to the center of the road:

![centered only training](./assets/track1-centered.gif)

The video above shows the car drives almost perfectly, however at a steep curve almost at the end it is unable to steer itself back to the center of the road.

Udacity suggested to record *recovering* driving behavior showing how to steer back to the center from a bad position.

Unfortunately the simulator makes it a bit difficult for one person alone to selectively record fragments of driving behavior, especially if using a gamepad: the simulator requires you to start/stop recording with the mouse while you control the car with a gamepad. I decided to use a different approach: record *bad behavior* and modify it in the same way the left/right cameras are adjusted to be useful as training material.

##### Learning from bad behavior: Sidewalk driving

I recorded left-sided driving: driving as best as possible with the car positioned as close as possible to the left line:

![Left-sided driving](./assets/left-LCR.gif "Left-sided driving")

Conversely for right-sided driving:

![Right-sided driving](./assets/right-LCR.gif "Right-sided driving")

For the left-sided driving I used the center and right cameras, both corrected to point to the center of the road with a steering bias of +0.5. The right-sided driving I used the center and left cameras with the equivalent correcting bias. As in the case of the left and right cameras, the correction bias indicates how quickly we want the car to steer itself back towards the center:

```
left_log     = read_log(args.leftdir,   t = "c r"  , cb =  0.5)
right_log    = read_log(args.rightdir,  t = "c l",   cb = -0.5)
```

##### Learning from more bad behavior: Skewed driving

Training the network with centered and sidewalk driving may be sufficient to get out from any position in which the car faces ahead; the training data  contains images where the car facing is parallel to the road. However should the car face the edge of the road in an oblique angle, correction may prove more difficult.

I recorded skewed-driving: driving from the center of the road trying to get out of the road, from center to the left:

![Skewed-left driving](./assets/skewed-left-LCR.gif "Skewed-left driving")

and from center to the right:

![Skewed-right driving](./assets/skewed-right-LCR.gif "Skewed-right driving")

In the same vein as as above, the skewed driving must be loaded with a steering correction bias, in this case I went with 0.7. Again, the bigger the value the quicker the car will steer back to the center as long as the overall end-to-end performance of the simulator and prediction pipeline is decent.

```
sk_left_log  = read_log(sk_left_dir,    t = "c l r", cb =  0.7)
sk_right_log = read_log(sk_right_dir,   t = "c l r", cb = -0.7)
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
