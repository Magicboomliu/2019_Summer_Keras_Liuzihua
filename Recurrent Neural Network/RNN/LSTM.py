__author__ = "Luke Liu"
#encoding="utf-8"
from keras.datasets import imdb

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_features = 10000

maxlen = 500

(input_train,y_train),(input_test,y_test) = imdb.load_data(
    num_words=max_features
)

input_train = sequence.pad_sequences(input_train,maxlen=maxlen)
input_test = sequence.pad_sequences(input_test,maxlen=maxlen)

from keras.models import  Sequential
from keras.layers import Dense,LSTM,Dropout,Conv2D,Embedding

# start the model
model = Sequential()
model.add(Embedding(input_dim=max_features,output_dim=32,input_length=maxlen))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.summary()
# in the compile, we need to tell others what is the loss function,and the optimater
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']
            )
hostry = model.fit(
    input_train,y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
)

