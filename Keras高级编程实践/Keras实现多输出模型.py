__author__ = "Luke Liu"
#encoding="utf-8"

# inputs are the social_media words of someone, what we want to do,
# is by knowing these data, we can predict the gender,the income level,and the age

# so the gender and the income are classification,and the age is regression
from  keras import Input
from  keras import Model
from keras.optimizers import RMSprop
from keras import  layers

vocabulary_size = 50000
num_income_groups = 10

input_text = Input(shape=(None,),dtype='int32',name='media_resource')

input_text_embedd = layers.Embedding(vocabulary_size,32,input_length=100)(input_text)
x = layers.Conv1D(128, 5, activation='relu')(input_text_embedd)
x=layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_pred = layers.Dense(1,name='age')(x)
income_pred=layers.Dense(10,activation='softmax',name='income')(x)
gender_pred=layers.Dense(1,activation='sigmoid',name='gender')(x)

model=Model(input_text,[age_pred,income_pred,gender_pred])
model.summary()

model.compile(optimizer='rmsprop',
              loss={
                  'age':'mse',
                  'income':'categorical_crossentropy',
                  'gender':'binary_crossentropy'
              },
              loss_weights={'age': 0.25, 'income': 1., 'gender': 10.}
              )
import numpy as np
input_text = np.random.randint(1,vocabulary_size,size=(5000,100))

# income
income_tr=np.random.randint(1,10,size=(5000,10))
# gender
gender_tr=np.random.randint(0,2,size=(5000,))

# age
age_tr=np.random.randint(10,70,size=(5000,))

model.fit(input_text,[age_tr,income_tr,gender_tr],
          batch_size=64,
          epochs=10,
          validation_split=0.2)