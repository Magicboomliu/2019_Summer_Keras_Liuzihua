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
from  keras.models import load_model
model = load_model("GRU_predict_tem.md5")
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


lookback = 1440
step = 6
delay = 144

dim_V= lookback//step
print(dim_V)

# if we want to predict the line 3000

orginal_data = np.array(lines[300000].split(",")[1:]).astype('float32')

# data process
precessed_data = (orginal_data-mean)/ std
print(precessed_data)

float_data_test = np.zeros((240,14))
for i,j in enumerate(range(300000-lookback,300000,step)):
    float_data_test[i,:]+= np.array(lines[j].split(",")[1:]).astype("float32")

true_data = np.array(lines[300000+144].split(",")[1]).astype("float32")
float_data_test=float_data_test.reshape((1,240,14))

pred = model.predict(float_data_test)
print(pred[0][0]*std[0]+mean[0])
print(true_data)
