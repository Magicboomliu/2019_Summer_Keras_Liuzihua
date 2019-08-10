__author__ = "Luke Liu"
#encoding="utf-8"
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
path='../dataset/aclimdb'
labels=[]
texts=[]
training_dir = os.path.join(path,'test')
for label_type in ['neg','pos']:
    ppath = os.path.join(training_dir,label_type)
    for file_name in os.listdir(ppath):
        if file_name[-4:]== '.txt':
            with open(os.path.join(ppath,file_name),'rb') as f1:
                texts.append(f1.read())
            if label_type=='neg':
                labels.append(0)
            else:
                labels.append(1)
Imdn_dict={"texts":texts,"comment":labels}
import  pickle
with open("IMnd_python_test",'wb') as f1:
    pickle.dump(Imdn_dict, f1)


