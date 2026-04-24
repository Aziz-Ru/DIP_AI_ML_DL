import tensorflow as tf
from keras.layers import Dense,Flatten,Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.applications import VGG16

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train[...,None]
x_test = x_test[...,None]

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))

def process_data(img,lbl):
    img = tf.image.resize(img,(224,224))
    img = tf.image.grayscale_to_rgb(img)
    img /=255.0
    return img,lbl

trainds = train_ds.map(process_data).batch(32)
testds  = test_ds.map(process_data).batch(32)

vgg_base = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))

for layer in vgg_base.layers:
    layer.trainable = False

x = Flatten()(vgg_base.output)
x= Dense(1024,activation = 'relu')(x)
x= Dense(512,activation = 'relu')(x)
output = Dense(10,activation = 'softmax')(x)

model = Model(inputs=vgg_base.input,outputs=output)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

tm = model.fit(trainds,epochs=2)

loss,acc=model.evaluate(testds)

print("accuracy"+acc)


