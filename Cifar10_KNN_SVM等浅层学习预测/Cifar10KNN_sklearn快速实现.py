__author__ = "Luke Liu"
#encoding="utf-8"

import numpy as np
import  os
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

datasets = 'D:/BaiduYunDownload/python_exe/dataset/cifar-10-batches-py'
datalsit = os.listdir(datasets)
data01 = unpickle(os.path.join(datasets, 'data_batch_1'))
data02 = unpickle(os.path.join(datasets, 'data_batch_2'))
data03 = unpickle(os.path.join(datasets, 'data_batch_3'))
data04 = unpickle(os.path.join(datasets, 'data_batch_4'))
data05 = unpickle(os.path.join(datasets, 'data_batch_5'))
labels_names_list = unpickle(os.path.join(datasets, 'batches.meta'))
# each data_batch is (10000 * 3072), which means (10000 * 32 *32 *3)
testing_data = unpickle(os.path.join(datasets, 'test_batch'))

# this is the data
X_trianing_data = np.concatenate((data01[b'data'], data02[b'data'], data03[b'data'], data04[b'data'], data05[b'data']),
                                 axis=0)
Y_trianing_label = np.concatenate(
    (data01[b'labels'], data02[b'labels'], data03[b'labels'], data04[b'labels'], data05[b'labels']), axis=0)
Y_training_label = np.array(Y_trianing_label)

X_testing_data = testing_data[b'data']
Y_testing_label = testing_data[b'labels']
Y_testing_label = np.array(Y_testing_label)

if __name__=='__main__':
    from  sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier (n_neighbors=5)
    knn.fit(X_trianing_data,Y_trianing_label)
    result= knn.predict(X_testing_data[:100])
    from sklearn import metrics
    scores= metrics.accuracy_score(result,Y_testing_label[:100])
    print("Accuracy Rate is :",scores)
