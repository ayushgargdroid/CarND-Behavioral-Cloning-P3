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
epochs = 4

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
    images = np.zeros([batch_size,test_img.shape[0]*3,test_img.shape[1],test_img.shape[2]])
    steering = np.zeros(batch_size)
    while True:
        index = counter = 0
        images = np.zeros([batch_size,test_img.shape[0]*3,test_img.shape[1],test_img.shape[2]])
        steering = np.zeros(batch_size)
        for i in np.random.permutation(range(size)):
            image1 = cv2.imread('./data/'+center_name[i])
            image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2YUV)

            image2 = cv2.imread('./data/'+left_name[i])
            image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2YUV)

            image3 = cv2.imread('./data/'+right_name[i])
            image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2YUV)
            steering[index] = steering_list[i]
            
            images[index] = np.vstack((image1,image2,image3))

            index += 1
            if(index == batch_size):
                counter += 1
                index = 0
                yield images,steering
                images = np.zeros([batch_size,test_img.shape[0]*3,test_img.shape[1],test_img.shape[2]])
                steering = np.zeros(batch_size)
            if(counter == int(size/batch_size)):
                break

# model = Sequential()
# model.add(Cropping2D(cropping=((60,25),(0,0)),input_shape=(160,320,3)))
# model.add(Lambda(lambda x: (x-127.5)/127.5,input_shape=(75,320,3)))
# model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
# model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
# model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
# model.add(Conv2D(64, 3, 3, activation='elu'))
# model.add(Conv2D(64, 3, 3, activation='elu'))
# model.add(Dropout(0.7))
# model.add(Flatten())
# model.add(Dense(100, activation='elu'))
# model.add(Dense(50, activation='elu'))
# model.add(Dense(10, activation='elu'))
# model.add(Dense(1))
# model.summary()

input_imgs = Input(shape=(160*3,320,3))

ttt_tower5 = Lambda(lambda x: K.slice(x, (0,0,0,0), (-1,160,320,3)),input_shape=(160,320,3))(input_imgs)

ttt_tower5 = Cropping2D(cropping=((60,25),(0,0)),input_shape=(160,320,3))(ttt_tower5)
ttt_tower5 = Lambda(lambda x: (x-127.5)/127.5)(ttt_tower5)
ttt_tower5 = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(ttt_tower5)
ttt_tower5 = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(ttt_tower5)
ttt_tower5 = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(ttt_tower5)
ttt_tower5 = Conv2D(64, (3, 3), activation='elu')(ttt_tower5)
ttt_tower5 = Conv2D(64, (3, 3), activation='elu')(ttt_tower5)

# ttt_tower[0] = Lambda(lambda x: x[160:320,:,:,:])(input_imgs)
ttt_tower6 = Lambda(lambda x: K.slice(x, (0,160,0,0), (-1,160,-1,-1)))(input_imgs)
ttt_tower6 = Cropping2D(cropping=((60,25),(0,0)),input_shape=(160,320,3))(ttt_tower6)
ttt_tower6 = Lambda(lambda x: (x-127.5)/127.5)(ttt_tower6)
ttt_tower6 = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(ttt_tower6)
ttt_tower6 = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(ttt_tower6)
ttt_tower6 = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(ttt_tower6)
ttt_tower6 = Conv2D(64, (3, 3), activation='elu')(ttt_tower6)
ttt_tower6 = Conv2D(64, (3, 3), activation='elu')(ttt_tower6)

# ttt_tower[0] = Lambda(lambda x: x[320:,:,:,:])(input_imgs)
ttt_tower7 = Lambda(lambda x: K.slice(x, (0,320,0,0), (-1,160,-1,-1)))(input_imgs)
ttt_tower7 = Cropping2D(cropping=((60,25),(0,0)),input_shape=(160,320,3))(ttt_tower7)
ttt_tower7 = Lambda(lambda x: (x-127.5)/127.5,input_shape=(75,320,3))(ttt_tower7)
ttt_tower7 = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(ttt_tower7)
ttt_tower7 = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(ttt_tower7)
ttt_tower7 = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(ttt_tower7)
ttt_tower7 = Conv2D(64, (3, 3), activation='elu')(ttt_tower7)
ttt_tower7 = Conv2D(64, (3, 3), activation='elu')(ttt_tower7)

# print(ttt_tower)
output = Concatenate(axis=1)([ttt_tower5,ttt_tower6,ttt_tower7])
print(output)
output = Flatten()(output)

output = Dense(300, activation='elu')(output)
output = Dense(150, activation='elu')(output)
output = Dense(75, activation='elu')(output)
output = Dense(10, activation='elu')(output)
out = Dense(1, activation='elu')(output)

model = Model(inputs = input_imgs, outputs = out)
model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

model.fit_generator(batch_generator(x_train[:,0],x_train[:,1],x_train[:,2],y_train,batch_size), steps_per_epoch=x_train.shape[0]//batch_size, epochs=epochs, verbose=1, validation_data=batch_generator(x_valid[:,0],x_valid[:,1],x_valid[:,2],y_valid,batch_size), validation_steps=(x_valid.shape[0]//batch_size))

# model.fit_generator(batch_generator(x_train[:,0],x_train[:,1],x_train[:,2],y_train,batch_size), steps_per_epoch=x_train.shape[0]//batch_size, epochs=epochs, verbose=1)


model.save('model2.h5')