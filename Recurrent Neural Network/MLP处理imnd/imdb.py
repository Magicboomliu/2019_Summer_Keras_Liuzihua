__author__ = "Luke Liu"
#encoding="utf-8"
from keras.datasets import imdb
from keras.preprocessing import  sequence  #用于截长补短
from keras.preprocessing.text import Tokenizer #建立字典

max_features = 10000  # 提取前10000个单词
maxlen = 20   #每个评论最多有20个单词
# here the shape of X_train is（2500，）,and the shape of y_train is (2500,)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

#将数字列表转化为向量列表（n_samples,maxlen)的 2-d matrix
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding,Dropout
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))  #将其embed
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(256,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
model.save("imdn_sample.md5")

