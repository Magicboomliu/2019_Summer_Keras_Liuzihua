__author__ = "Luke Liu"
#encoding="utf-8"

# bulid a  new model
from keras import Input
from keras import Model
from  keras import  layers

# each dim has length
text_vocabulary_size = 10000
question_vocabulary_size =10000
answer_vocabulary_size = 5000

test_input = Input(shape=(None,),dtype='int32',name='test')

embedd_text = layers.Embedding(text_vocabulary_size,64)(test_input)
encode_text = layers.LSTM(32)(embedd_text)

question_input = Input(shape=(None,),dtype= 'int32',name='question')
embedd_question = layers.Embedding(question_vocabulary_size,32)(question_input)
encode_question = layers.LSTM(16)(embedd_text)

concatenated = layers.concatenate([encode_text, encode_question],
                                  axis=-1)
answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated)

model= Model([test_input,question_input],answer)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

# train the model
import  numpy as np
# here is tha training data
num_samples=1000
max_len  = 100
test = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_len))
question = np.random.randint(1, question_vocabulary_size,
                             size=(num_samples, max_len))
answer=np.random.randint(1,answer_vocabulary_size,size=(num_samples))
from keras.utils import np_utils
answer_Onehot = np_utils.to_categorical(answer,answer_vocabulary_size)

# Also some validation data
num_samples_vali=80
max_len  = 100
validation_test = np.random.randint(1,text_vocabulary_size,size=(num_samples_vali,max_len))
question_vali = np.random.randint(1,question_vocabulary_size,size=(num_samples_vali,max_len))
answer_vali=np.random.randint(1,answer_vocabulary_size,size=(num_samples_vali))
from keras.utils import np_utils
answer_Onehot_vali = np_utils.to_categorical(answer_vali,answer_vocabulary_size)


model.fit([test,question],answer_Onehot,
          epochs=10,batch_size=128,
         validation_data=[[validation_test,question_vali],answer_Onehot_vali]
          )
# model.fit({'test':test,'question':question},answer,epochs=10,batch_size=128)