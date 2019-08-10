__author__ = "Luke Liu"
#encoding="utf-8"
from keras import backend as K
from  keras.applications import VGG16

conv_base = VGG16(weights='imagenet',include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = conv_base.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])
grads = K.gradients(loss, conv_base.input)[0]
# 标准化梯度
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([conv_base.input],[loss,grads])
import numpy as np
# 灰度噪声图片
input_image_data = np.random.random((1,150,150,3)) * 20+ 128
# GD process
import matplotlib.pyplot as plt
step=1
for i in range(40):
    loss_value,grads_vale = iterate([input_image_data])
    input_image_data += grads_vale * step

img= input_image_data[0]

plt.imshow(img/255.)
plt.show()
