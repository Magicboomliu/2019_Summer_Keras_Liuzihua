__author__ = "Luke Liu"
#encoding="utf-8"
import keras
from  keras.utils import  np_utils
import  matplotlib.pyplot as plt
import  cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
dataset='Cifa10_python'

if __name__=="__main__":
    cifa_data= unpickle(dataset)
    X_image_train = cifa_data['X_train']
    X_image_test = cifa_data['X_test']
    y_train_label = cifa_data['y_train']
    y_test_label = cifa_data['y_test']

    #X预处理
    X_image_train_normalize = X_image_train.astype('float32')/255.0
    X_image_test_normalize = X_image_test.astype('float32')/255.0
    #Y预处理
    y_train_label_OneHot = np_utils.to_categorical(y_train_label)
    y_test_label_OneHot = np_utils.to_categorical(y_test_label)

    #建立 Model

    from keras.models import Sequential
    from keras.layers import Dense,Dropout,Activation,Flatten
    from keras.layers import Conv2D,MaxPool2D,ZeroPadding2D

    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),
                     input_shape=(32,32,3),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(rate=0.25))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),
                     activation='relu',
                     padding='same'))

    model.add(Dropout(rate=0.25))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dropout(rate=0.25))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(rate=0.25))

    model.add(Dense(10,activation='softmax'))

    # print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    train_history = model.fit(X_image_train_normalize,y_train_label_OneHot,
                              validation_split=0.2,
                              epochs=10,batch_size=120,verbose=1)
    scores= model.evaluate(X_image_test_normalize,y_test_label_OneHot,verbose=0)
    print(scores[1])
    model.save('cifa10_model.h5')
