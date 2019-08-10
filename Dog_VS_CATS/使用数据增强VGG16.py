__author__ = "Luke Liu"
#encoding="utf-8"
from  keras.applications import  VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
import os
import  pandas
import cv2
# 这是数据分类
dataset_training = 'dataset/training_set/training_data'
dataset_validation= 'dataset/training_set/validation_data'
dataset_testing='dataset/test_set'
# Data Image Classification
cat_train = os.path.join(dataset_training,'cat_training')
dog_train = os.path.join(dataset_training,'dog_training')
cat_validation=os.path.join(dataset_validation,'cat_validation')
dog_validation=os.path.join(dataset_validation,'dog_validation')
cat_test=os.path.join(dataset_testing,'cats')
dog_test = os.path.join(dataset_testing,'dogs')

#构建分类模型
from  keras.models import Sequential
from keras import optimizers
from keras.layers import Dense,Dropout,Conv2D,Flatten
from keras_preprocessing.image import   ImageDataGenerator


model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
conv_base.trainable=False

datagen=ImageDataGenerator(rescale=1./255,
                           rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                           zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

training_generator = datagen.flow_from_directory(
    dataset_training,
    batch_size=20,
    target_size=(150,150),
    class_mode='binary'
)
validation_generator = datagen.flow_from_directory(
    dataset_validation,
    batch_size=20,
    target_size=(150,150),
    class_mode='binary'

)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

model.fit_generator(training_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50)

model.save("Inhance_VGG16_CD.md5")