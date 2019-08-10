__author__ = "Luke Liu"
#encoding="utf-8"
from keras.models import Model,Sequential
from  keras import  layers
from keras import  Input
from keras import  activations
input_shape = Input((64,))
x=layers.Dense(32,activation='relu')(input_shape)
x=layers.Dense(64,activation='relu')(x)
x=layers.Dense(64,activation='relu')(x)
x=layers.Dense(128,activation='relu')(x)
output_shape = layers.Dense(1,activation='sigmoid')(x)

model = Model(input_shape,output_shape)
model.summary()
