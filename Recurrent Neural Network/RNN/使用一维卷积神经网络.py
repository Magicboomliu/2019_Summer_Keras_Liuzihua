__author__ = "Luke Liu"
#encoding="utf-8"
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,Dense,LSTM
from  keras.preprocessing import  sequence
from  keras.datasets import  imdb
max_features=10000
maxlen = 500
(x_train,y_train),(x_test,y_test) =imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train,500)
x_test =  sequence.pad_sequences(x_test,500)


from keras.models import  Sequential
from keras.layers import  Dense,Conv1D,MaxPooling1D,Embedding,GlobalMaxPooling1D


from keras.optimizers import RMSprop

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Conv1D(32, 7, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)