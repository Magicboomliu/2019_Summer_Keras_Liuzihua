__author__ = "Luke Liu"
#encoding="utf-8"

from  keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#conv_base.summary()

import  os
from keras import models,layers,Sequential,optimizers
import  numpy as np
from  keras.preprocessing.image import  ImageDataGenerator

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

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_featuers(directory,n_samples):
    features= np.zeros(shape=(n_samples,4,4,512))
    labels = np.zeros(shape=(n_samples))
    generator= datagen.flow_from_directory(
        directory,
        batch_size=batch_size,
        target_size=(150,150),
        class_mode="binary",

    )
    i=0
    for input_batches,label_batches in generator:
        feature_batches = conv_base.predict(input_batches)
        features[i*batch_size:(i+1)*batch_size] = feature_batches
        labels[i*batch_size:(i+1)*batch_size] = label_batches
        i+=1
        if i* batch_size >= n_samples:
            break
    return features,labels

training_features ,train_labels = extract_featuers(dataset_training,6000)
validation_features, validation_labels = extract_featuers(dataset_validation,2000)
testing_features, testing_labels =extract_featuers(dataset_testing,2020)

# Flatten处理
train_features = np.reshape(training_features, (6000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (2000, 4 * 4 * 512))
test_features = np.reshape(testing_features, (2020, 4 * 4 * 512))

from  keras.layers import Dense,Dropout
#去建立紧密连接层
model=Sequential()
model.add(Dense(266,activation='relu',input_dim=4*4*512))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

#compile the model

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

#train the model
history = model.fit(train_features,train_labels,
                    batch_size=20,
                    epochs=30,
                    validation_data=(validation_features,validation_labels)
                    )
model.save("VGG16plusDense_Dogs_Cats.md5")
