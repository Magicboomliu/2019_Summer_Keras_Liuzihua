__author__ = "Luke Liu"
#encoding="utf-8"
# 这是数据分类
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

print("training data count",len(os.listdir(cat_train))+len(os.listdir(dog_train)))
print("validation data count",len(os.listdir(cat_validation))+len(os.listdir(dog_validation)))
print("testing data count",len(os.listdir(cat_test))+len(os.listdir(dog_test)))

from keras.preprocessing.image import  ImageDataGenerator
datagen=ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    directory=dataset_validation,
    batch_size=20,
    class_mode='binary',
    target_size=(150,150)
)