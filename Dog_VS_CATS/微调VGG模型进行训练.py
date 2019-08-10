__author__ = "Luke Liu"
#encoding="utf-8"
from keras.applications import  VGG16
import os
from  keras.models import  Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras import optimizers,metrics
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
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# 微调网络
conv_base.trainable= True
set_trainable = False
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        set_trainable = True
    if set_trainable ==True:
        layer.trainable=True
    else:
        layer.trainable=False
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(1./255)

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

model=Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
    training_generator,
    epochs=30,
    steps_per_epoch=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save("SmallAdjust_model.md5")