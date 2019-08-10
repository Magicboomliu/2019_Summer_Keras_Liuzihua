__author__ = "Luke Liu"
#encoding="utf-8"
import matplotlib.pyplot as plt
import  cv2
import  keras
import os
from  keras.models import  load_model
dicts = {0:"Cat",1:"Dog"}
def changeSize(image_src):
    import cv2
    im1=cv2.imread(image_src)
    im1 = im1.astype('float32')/255.0
    im1=cv2.resize(im1,(150,150))
    im1 = im1.reshape((1,150,150,3))
    return im1
def show_image(src,pred):
    import  cv2
    im1=cv2.imread(src)
    plt.imshow(im1)
    plt.title("Prediction Value Is {}".format(pred))
    plt.xticks([])
    plt.yticks([])
    plt.show()

predict_path = 'Dogs_Cat_model.md5'
model = load_model(predict_path)
dataset_testing='dataset/test_set'
cat_test=os.path.join(dataset_testing,'cats')
dog_test = os.path.join(dataset_testing,'dogs')
#
# label_predict = model.predict_classes(changeSize(os.path.join(cat_test,os.listdir(cat_test)[77])))
#
# show_image(os.path.join(cat_test,os.listdir(cat_test)[77]),dicts[label_predict[0][0]])

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator=test_datagen.flow_from_directory(
    dataset_testing,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
scrore1 = model.evaluate_generator(test_generator)
model2 = load_model("Dogs_Cat_model_inhance.md5")
score2 = model2.evaluate_generator(test_generator)
model3=load_model("SmallAdjust_model.md5")
score3 = model3.evaluate_generator(test_generator)
print("未使用数据增强的结果：",scrore1[1])
print("使用数据增强的结果：",score2[1])
print("微调VGG16的结果：",score3[1])

# from keras.applications import  VGG16
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# pic=os.path.join(cat_test,os.listdir(cat_test)[77])
# pic_array = changeSize(pic)
# input_image= conv_base.predict(pic_array)
# inputs = input_image.reshape(1,4*4*512)
# VGG16_dense_model = load_model("VGG16plusDense_Dogs_Cats.md5")
# predd = VGG16_dense_model.predict_classes(inputs)
#
# VGG16_inhance = load_model("Dogs_Cat_model_inhance.md5")
# preddd= VGG16_inhance.predict_classes(changeSize(pic))
# show_image(pic,dicts[preddd[0][0]])
# print(dicts[preddd[0][0]])