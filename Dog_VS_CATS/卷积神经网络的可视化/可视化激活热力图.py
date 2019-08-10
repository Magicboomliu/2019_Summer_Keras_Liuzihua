__author__ = "Luke Liu"
#encoding="utf-8"
#这种通用的技术叫作类激活图（CAM，class activation map）可视化，它是指对输入图像生 成类激活的热力图。
# 类激活热力图是与特定输出类别相关的二维分数网格，对任何输入图像的 每个位置都要进行计算，它表示每个位置对该类别的重要程度。
# 举例来说，对于输入到猫狗分 类卷积神经网络的一张图像，CAM 可视化可以生成类别“猫”的热力图，
# 表示图像的各个部分 与“猫”的相似程度，CAM 可视化也会生成类别“狗”的热力图，表示图像的各个部分与“狗” 的相似程度。
from  keras.applications import VGG16
model = VGG16(weights='imagenet',include_top=False)
model.summary()
