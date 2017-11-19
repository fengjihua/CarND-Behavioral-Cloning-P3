# **Behavioral Cloning** 

### Document about how to clone human's driving behavior with neural network

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_cpu.json containing architecture of convolution neural network 
* model.h5 containing a trained convolution neural network (including architecture and weights)
* run1.mp4 video of running a loop with convolution neural network
* P3_v18.ipynb jupyter file of model
* P3_v18.html jupyter html report of model
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Note that I changed some code in the original drive.py:

1. Change speed to 30.
2. Comment 'load_model' line. (line 122) (Because load_model always got unknown error on my local machine)
3. First I loaded architecture from a model_cpu.json, then loaded weights from model.h5. (line 124~131)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

This is my model architecture:
1. First of all, crop image layer to crop the region car didn't insterest in, it could also reduce the calculation.
2. Image data normalized to range [-0.5, 0.5] in the model using a Keras lambda layer.
3. I used strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.
4. RELU layers to activate the data
5. Dropout layer after five convolution layers in order to reduce over-fitting
6. Three fully connected layers followed by the flatten layer leading to an output control value which is the inverse turning radius. The fully connected layers are designed to function as a controller for steering.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce over-fitting (model.py lines 119). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 62-68). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 241).

Train 5 epochs is a good choice. More than 10 epochs training will be over-fitting.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Here is my training strategy detail:

1. I used a combination of center and center flip images to train the model, then got a 'straight way only' driver, but it had a really good skill on straight line.
2. I added left, right images and left right flip images with a angle factor 0.05~0.2 to train the model, then got a 'drunk' driver, it could finish a loop without a crash but really made me sick.
3. I tried many times with differet angle factors and different combination of center left right images. Finally found center images made me a good skill driver but always crashed on turn way, and combination of left right images made me a drunk driver but never crashed.
4. Then I decided to use center and center flip images in each training batch to make a good skill driver (model.py line 188~196)
5. And I used left right and left right flip images in turn way batch to make a non-crash driver (model.py line 199~228)
6. After several times tuning, good skill driver plus non-crash driver equals perfect driver
7. Finally, I added more angle factors on steer values, then I got a perfect racer, pretty cool!!! (model.py line 211~221)