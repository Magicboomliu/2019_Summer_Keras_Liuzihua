__author__ = "Luke Liu"
#encoding="utf-8"
from keras.models import  load_model
import os
model = load_model("../Dogs_Cat_model.md5")
model.summary()

path = '../inhance'
pic_list = os.listdir(path)
img_path = os.path.join(path,pic_list[0])

import  matplotlib.pyplot as  plt
from keras.preprocessing import  image
import  numpy as np
img = image.load_img(img_path,target_size=(150,150))
img_tensor= image.img_to_array(img)
img_tensor =np.expand_dims(img_tensor,axis=0)
img_tensor=img_tensor/255.
# plt.imshow(img_tensor[0])
# plt.xticks([])
# plt.yticks([])
# plt.show()
from keras import layers
from keras import  models
layers_output=[layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.inputs,outputs= layers_output)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[3]

print(first_layer_activation.shape)
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(first_layer_activation[0,:,:,i])
plt.show()

# 随着层数的不断增加，激活出的结果放映的图像信息越来越少，相反的，反应的类别信息则会越来越多
# 深度学习网路又被称之为信息蒸馏管道
# 和人感知世界的原理是一样的

