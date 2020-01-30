import sys
sys.path = sys.path[2:]
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
import math
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import os
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Input, Concatenate
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam, SGD

learning_rate = 0.001
batch_size = 32
epochs = 6

center_imgs_name = []
left_imgs_name = []
right_imgs_name = []
steering = []

with open('./data/driving_log.csv', mode='r') as infile:
    reader = csv.reader(infile)
    i = 0
    for rows in reader:
        if i is 0:
            i+=1
            continue
        center_imgs_name.append(rows[0])
        left_imgs_name.append(rows[1][1:])
        right_imgs_name.append(rows[2][1:])
        steering.append(float(rows[3]))

steering = np.array(steering)
center_imgs_name = np.array(center_imgs_name)
left_imgs_name = np.array(left_imgs_name)
right_imgs_name = np.array(right_imgs_name)
x_train = np.vstack((center_imgs_name,left_imgs_name,right_imgs_name)).transpose()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, steering,test_size = 0.4)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size = 0.5)

def batch_generator(center_name,left_name,right_name,steering_list,batch_size):
    print(center_name.shape)
    test_img = cv2.imread('./data/'+center_name[0])
    size = center_name.shape[0]
    index = 0
    counter = 0
    images = np.zeros([batch_size,test_img.shape[0],test_img.shape[1],test_img.shape[2]])
    steering = np.zeros(batch_size)
    while True:
        index = counter = 0
        images = np.zeros([batch_size,test_img.shape[0],test_img.shape[1],test_img.shape[2]])
        steering = np.zeros(batch_size)
        for i in np.random.permutation(range(size)):
            t_prob = np.random.rand()
            if(t_prob < 0.33):
                image = cv2.imread('./data/'+center_name[i])
                steering[index] = steering_list[i]
            elif(t_prob < 0.66):
                image = cv2.imread('./data/'+left_name[i])
                steering[index] = steering_list[i] + 0.2
            else:
                image = cv2.imread('./data/'+right_name[i])
                steering[index] = steering_list[i] - 0.2
                
            if(np.random.rand() < 0.6):
                image = cv2.flip(image,1)
                steering[index] = -steering[index]
            
            images[index] = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)

            index += 1
            if(index == batch_size):
                counter += 1
                index = 0
                yield images,steering
                images = np.zeros([batch_size,test_img.shape[0],test_img.shape[1],test_img.shape[2]])
                steering = np.zeros(batch_size)
            if(counter == int(size/batch_size)):
                break

model = Sequential()
model.add(Cropping2D(cropping=((60,25),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x-127.5)/127.5,input_shape=(75,320,3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

model.fit_generator(batch_generator(x_train[:,0],x_train[:,1],x_train[:,2],y_train,batch_size), steps_per_epoch=x_train.shape[0]//batch_size, epochs=epochs, verbose=1, validation_data=batch_generator(x_valid[:,0],x_valid[:,1],x_valid[:,2],y_valid,batch_size), validation_steps=(x_valid.shape[0]//batch_size))

# model.fit_generator(batch_generator(x_train[:,0],x_train[:,1],x_train[:,2],y_train,batch_size), steps_per_epoch=x_train.shape[0]//batch_size, epochs=epochs, verbose=1)


model.save('model4.h5')