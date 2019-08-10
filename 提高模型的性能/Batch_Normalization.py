__author__ = "Luke Liu"
#encoding="utf-8"
from keras import layers
from keras.applications import VGG16
from keras.models import Sequential
conv_model=Sequential()
conv_model.add(layers.Conv2D(32, 3, activation='relu'))
# 加入这一层 BN
conv_model.add(layers.BatchNormalization())

dense_model=Sequential()
dense_model.add(layers.Dense(32, activation='relu'))
# 加入这一层 BN
dense_model.add(layers.BatchNormalization())