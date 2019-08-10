__author__ = "Luke Liu"
#encoding="utf-8"

# 使用了ensembling的思想
from keras import models
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,Dense,LSTM
from  keras.preprocessing import  sequence
from  keras.datasets import  imdb
max_features=10000
maxlen = 500
(x_train,y_train),(x_test,y_test) =imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train,500)
x_test =  sequence.pad_sequences(x_test,500)

model = Sequential()
model.add(Embedding(max_features, 32,input_length=maxlen))
# avoid overfitting
model.add(Bidirectional(LSTM(32,dropout=0.2,
                             recurrent_dropout=0.25,
                             )))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# 过去的数据，或是最近过去的数据，能够比较好的预测未来