import sys
sys.path = sys.path[2:]
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
import math
import numpy as np
import csv
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

learning_rate = 0.001
batch_size = 128
epochs = 5

center_imgs_name = []
steering = []

with open('./data/driving_log.csv', mode='r') as infile:
    reader = csv.reader(infile)
    i = 0
    for rows in reader:
        if i is 0:
            i+=1
            continue
        center_imgs_name.append(rows[0])
        steering.append(float(rows[3]))

steering = np.array(steering)

def batch_generator(center_name,left_name,right_name,steering_list,operation):
    test_img = cv2.imread('./data/'+center_name[0])
    size = 0
    start = 0
    if(operation == 'train'):
        start = 0
        size = int(0.6*len(center_name))
    elif(operation == 'test'):
        start = int(0.8*len(center_name))
        size = int(0.2*len(center_name))
    elif(operation == 'valid'):
        start = int(0.6*len(center_name))
        size = int(0.2*len(center_name))
    elif(operation == 'viz'):
        start = int(np.random.rand()*len(center_name))
        size = 1
    images = np.empty([size,test_img.shape[0]-60-25,test_img.shape[1],test_img.shape[2]])
    steering = np.empty(size)
    index = 0
    for i in np.random.permutation(range(start,start+size)):
        image = cv2.imread('./data/'+center_name[i])[60:-25,:,:]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        images[index] = image
        steering[index] = steering_list[i]
    yield images,steering

tt = batch_generator(center_imgs_name,None,None,steering,'viz')
sample_img, _ = next(tt)
sample_img = np.uint8(sample_img[0])

model = Sequential()
model.add(Lambda(lambda x: (x-127.5)/127.5,input_shape=sample_img.shape))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
model.fit_generator(batch_generator(center_imgs_name,None,None,steering,'train'),batch_size,epochs,validation_data=batch_generator(center_imgs_name,None,None,steering,'valid'),nb_val_samples=0.2*len(center_imgs_name),verbose=1)
