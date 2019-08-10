__author__ = "Luke Liu"
#encoding="utf-8"
import  os

path = '../dataset'
dataset=os.path.join(path,os.listdir(path)[1])
f=open(dataset)
data = f.read()
f.close()
lines = data.split('\n')
# return the header list
header = lines[0].split(',')
# return he datas
lines = lines[1:]  # here is 420551 rows
print(header)
# for each data ,we do not want "Date Time"
import numpy as np
float_data=np.zeros((len(lines),len(header)-1))

for i,line in enumerate(lines):
    values= [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values
# import matplotlib.pyplot as  plt
# plt.plot(float_data[:,1])
# plt.show()
print(float_data.shape)
# 使用前200 000个数据作为训练数据（一共 420 551个数据）
# 对数据进行标准化处理
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
import numpy as np
def generator(data,lookback,delay,min_index,max_index,shuttle=False,batch_size=128,step=6):
    if max_index is None:
        max_index =len(data)- delay - 1
    i = min_index+lookback
    while 1:
        if shuttle:
            rows = np.random.randint(min_index,max_index,batch_size)
        else:
            if  i+batch_size>= max_index:
                i= min_index +lookback
            rows = np.arange(i,min(i+batch_size,max_index))
            i+= len(rows)
        samples = np.zeros((len(rows),
                            lookback//step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j,row in enumerate(rows):
            indices = range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]
        yield  samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128
# 前200 000 取样作为（n_samples,target）的数据
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuttle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)  //batch_size
test_steps = (len(float_data) - 300001 - lookback)  //batch_size
print(val_steps)
print(test_steps)

# 一种基于常识的非机器学习的方法，我们有理由认为明天的温度可能等于今天的温度
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()  # this is the standard of the model performance

# build the model

from keras.models import  Sequential
from keras.layers import Dense,Dropout,GRU,LSTM,SimpleRNN
from keras import optimizers
from keras import losses

model = Sequential()
model.add(GRU(32,input_shape=(lookback//step,14)))
model.add(Dense(1))

model.compile(optimizer=optimizers.RMSprop(),loss=losses.mae)
history = model.fit_generator(train_gen,
                    steps_per_epoch=500,
                    epochs=20,
                    validation_data=val_gen,
                    validation_steps= val_steps)

model.save("GRU_predict_tem.md5")

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


