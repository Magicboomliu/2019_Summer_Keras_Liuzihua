__author__ = "Luke Liu"
#encoding="utf-8"
import  numpy as np
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
print(token_index)


import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer

text1='some thing to eat'
text2='some thing to drink'
texts=[text1,text2]

print (T.text_to_word_sequence(text1))  #['some', 'thing', 'to', 'eat']
print (T.one_hot(text1,10) ) #[7, 9, 3, 4]
print (T.one_hot(text2,10)) #[7, 9, 3, 1]

tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(texts)

print (tokenizer.word_counts) #[('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)]
print (tokenizer.word_index) #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
print (tokenizer.word_docs) #{'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
print (tokenizer.index_docs) #{1: 2, 2: 2, 3: 2, 4: 1, 5: 1}

tokenizer.texts_to_sequences(texts) #[[1, 2, 3, 4], [1, 2, 3, 5]]

