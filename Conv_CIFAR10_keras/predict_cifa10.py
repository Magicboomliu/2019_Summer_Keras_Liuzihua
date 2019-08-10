__author__ = "Luke Liu"
#encoding="utf-8"

import keras
from  keras.utils import  np_utils
import  matplotlib.pyplot as plt
import  cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
from keras.models import  load_model
label_dict={0:"airplane",1:'automobile',2:"bird",3:"cat",4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:"truck"}
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def plot_image_labels_prediction(X_data,y_label,prediction,deer,end):
    import matplotlib.pyplot as plt
    show_nums = end-deer+1
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    if show_nums>25:
        show_nums=25
    for i in range(0,show_nums):
        ax =plt.subplot(5,5,i+1)
        ax=plt.imshow(X_data[i+show_nums+1])
        title=str(i)+ ": O: " + label_dict[y_label[i+show_nums+1][0]]+ " P: "+label_dict[prediction[i+show_nums+1]]
        plt.title(title,fontsize=8)
        plt.xticks([])
        plt.yticks([])

    plt.show()
def show_possibility_of_pic(image,no,Pro):
    import  matplotlib.pyplot as plt
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    for i in range(10):
        strs= label_dict[i]+"'s P is : "+ str(round(Pro[no][i]))
        plt.text(-10,0+i*4,strs,fontsize=6)
    plt.show()

dataset_path='Cifa10_python'
model = load_model('cifa10_model.h5')

dataset=unpickle(dataset_path)
X_image_test = dataset['X_test']
y_test_label = dataset['y_test']

#normalize & OneHot Code
X_image_test_normalize = X_image_test.astype('float32')/255.0
y_test_label_OneHot=np_utils.to_categorical(y_test_label)

y_predict = model.predict_classes(X_image_test_normalize)

# show the image
# plot_image_labels_prediction(X_image_test,y_test_label,y_predict,0,24)

Predict_Probability = model.predict(X_image_test_normalize)
# show the possibility of the Image
# show_possibility_of_pic(X_image_test[1],1,Predict_Probability)
# 建立混淆矩阵
import pandas as pd
x = pd.crosstab(y_test_label.reshape(-1),y_predict,
            rownames=['label'],colnames=['predict'])
print(x)