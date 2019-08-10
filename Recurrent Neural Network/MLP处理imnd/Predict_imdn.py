__author__ = "Luke Liu"
#encoding="utf-8"
from keras.preprocessing import  sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import  load_model
from keras.metrics import binary_accuracy,binary_crossentropy

model=load_model("imdn_sample.md5")
preview  ="Was lifeless, characters had no emotion or personality , found myself rooting for gaston, he was the only cool one. Modern autotuned garbage. CGI was awful and way overdone. Nothing like the original. Songs were long and annoying. Dialogue felt rushed like they tried to cram all the stuff from the original into it but ran out of time."
texts=[preview]
token = Tokenizer(num_words=10000)
token.fit_on_texts(texts)
#将影评文字转化为数字列表
intput_seq = token.texts_to_sequences(texts=texts)

#将数字列表转化为向量
pad_intput_seq = sequence.pad_sequences(intput_seq,maxlen=20)
a = token.word_index
print(a)