__author__ = "Luke Liu"
#encoding="utf-8"

import keras
from keras.utils import  np_utils
import  matplotlib.pyplot as plt
import  cv2
import os
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
if __name__=="__main__":
    from  keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen  = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        dataset_training,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        dataset_validation,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary'
    )

    # set up models
    from keras import layers
    from keras import models
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # print(model.summary())

    from keras import optimizers

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )

    model.save("Dogs_Cat_model.md5")
