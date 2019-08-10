__author__ = "Luke Liu"
#encoding="utf-8"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

import  os
import  numpy as np

class K_nearestNeighbour(object):
    def __init__(self):
        self.X = None
        self.Y =None
        self.dict= 0

    def train(self,x,y):
        self.xtr=x
        self.ytr=y


    def L1distance(self,x):
        dist = np.sum(np.abs(self.xtr-x),axis=1)
        return  dist
    def L2distance(self,x):
        dist = np.square(np.sum(self.xtr-x,axis=1))
        return dist
    def predict(self,test_X,k,distance):
        num_test = test_X.shape[0]
        pred = []
        for i in range(num_test):
            if distance=="L1":
                self.dis = self.L1distance(test_X[i])
            if distance=="L2":
                self.dis=self.L2distance(test_X[i])

            #排序，返回其下标
            disArgsort = np.argsort(self.dis)[0:k]
            # 根据下标返回labels 列表
            classArgsort= self.ytr[disArgsort]
            # 返回所有列表的每个元素的个数
            classargcount=np.bincount(classArgsort)
            # 找到最多的个数，对应的下标值的值正好是label
            predictClass = np.argmax(classargcount)

            pred.append(predictClass)
        return np.array(pred)

if __name__=="__main__":

    datasets = 'D:/BaiduYunDownload/python_exe/dataset/cifar-10-batches-py'
    datalsit= os.listdir(datasets)
    data01 = unpickle(os.path.join(datasets,'data_batch_1'))
    data02 = unpickle(os.path.join(datasets, 'data_batch_2'))
    data03 = unpickle(os.path.join(datasets, 'data_batch_3'))
    data04 = unpickle(os.path.join(datasets, 'data_batch_4'))
    data05 = unpickle(os.path.join(datasets, 'data_batch_5'))
    labels_names_list= unpickle(os.path.join(datasets,'batches.meta'))
    # each data_batch is (10000 * 3072), which means (10000 * 32 *32 *3)
    testing_data=unpickle(os.path.join(datasets,'test_batch'))

    # this is the data
    X_trianing_data = np.concatenate((data01[b'data'],data02[ b'data'],data03[b'data'],data04[b'data'],data05[b'data']),axis=0)
    Y_trianing_label = np.concatenate((data01[b'labels'],data02[b'labels'],data03[b'labels'],data04[b'labels'],data05[b'labels']),axis=0)
    Y_training_label = np.array(Y_trianing_label)

    X_testing_data = testing_data[b'data']
    Y_testing_label =testing_data[b'labels']
    Y_testing_label = np.array(Y_testing_label)

    knn_model = K_nearestNeighbour()
    print("Start training-------------------------")
    knn_model.train(X_trianing_data,Y_training_label)
    print("Start Predicting---------------------")
    pre = knn_model.predict(X_testing_data[:100],5,"L1")
    print("Finishing----------------------------")
    loss = Y_testing_label[:100] - pre[:100]
    cnt=0
    for i in loss:
        if i==0:
            cnt+=1
    acc= cnt*1.0/len(Y_testing_label[:100])
    print("Accuracy Rate is ",acc)









