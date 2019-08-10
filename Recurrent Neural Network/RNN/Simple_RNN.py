__author__ = "Luke Liu"
#encoding="utf-8"

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import  Tokenizer

maxlen=500
bacth_size=32
print("Loading the data..................................")
(input_train,y_train),(input_test,y_test) = imdb.load_data(
    num_words=2000
)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# build the model

from keras.layers import Dense
model = Sequential()
model.add(Embedding(2000, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
