__author__ = "Luke Liu"
#encoding="utf-8"
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

dataset_path = os.listdir(os.getcwd())[:2]
training_data_dir = os.path.join(os.getcwd(),dataset_path[0])
testing_data_dir = os.path.join(os.getcwd(),dataset_path[1])
def unpickle(dir):
    import pickle
    with open(dir, 'rb') as fo:
        dict = pickle.load(fo, encoding='utf-8')
    return dict
conver_dict = {0:"Bad",1:"Great"}
if __name__ =="__main__":
    training_data= unpickle(training_data_dir)
    testing_data = unpickle(testing_data_dir)
    # data classification
    input_training_data = training_data['texts']
    input_training_datas=[]
    for i in input_training_data:
        input_training_datas.append(str(i,encoding='utf-8'))
    y_trainig_data = training_data['comment']
    input_testing_data = testing_data['texts']
    input_testing_datas=[]
    for j in input_testing_data:
        input_testing_datas.append(str(j,encoding='utf-8'))
    y_testing_data =testing_data['comment']

    max_features=10000
    max_len = 500
    token  = Tokenizer(num_words=max_features)
    token.fit_on_texts(input_training_datas)
    # converge the text to seq
    input_training_data_sequences = token.texts_to_sequences(input_training_datas)
    input_testing_data_sequences = token.texts_to_sequences(input_testing_datas)
    print("Now is loading data,Please wait......")
    # converge the text to the maxlen shape
    input_train = sequence.pad_sequences(input_training_data_sequences,max_len)
    input_test = sequence.pad_sequences(input_testing_data_sequences,max_len)
    print("Loading data is over")

    #build the model
    from  keras.models import  Sequential
    model= Sequential()
    from keras.layers import  Dense,Embedding,LSTM,SimpleRNN,Dropout
    model.add(Embedding(input_dim=max_features,output_dim=32,input_length=max_len))
    model.add(Dropout(0.25))
    model.add(LSTM(32))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    from  keras import optimizers
    from keras import losses
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    history  =model.fit(input_train,y_trainig_data,
                        batch_size=128,
                        epochs=10,
                        validation_split=0.2
                        )
    model.save("RNN_LSTM_string.md5")
