__author__ = "Luke Liu"
#encoding="utf-8"
import keras
import  numpy as np
import os
np.random.seed(10)
#  this func used to load the data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def showpictures(arrays):
    channels_r=arrays[:1024]
    channels_g=arrays[1024:2048]
    channels_b=arrays[2048:]
    image_datas=np.column_stack((channels_r,channels_g,channels_b))
    im =image_datas.reshape((32,32,3))
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()

if __name__=="__main__":
    datasets = 'D:/BaiduYunDownload/python_exe/dataset/cifar-10-batches-py'
    datalsit= os.listdir(datasets)
    data01 = unpickle(os.path.join(datasets,'data_batch_1'))
    data02 = unpickle(os.path.join(datasets, 'data_batch_2'))
    data03 = unpickle(os.path.join(datasets, 'data_batch_3'))
    data04 = unpickle(os.path.join(datasets, 'data_batch_4'))
    data05 = unpickle(os.path.join(datasets, 'data_batch_5'))
    labels_names_list= unpickle(os.path.join(datasets,'batches.meta'))
    import  cv2
    import matplotlib.pyplot as plt
    training01_data = data01[b'data']
    training02_data = data02[b'data']
    training03_data = data03[b'data']
    training04_data = data04[b'data']
    training05_data = data05[b'data']
    training_data_orgin=np.vstack((training01_data,training02_data,training03_data,training04_data,training05_data))
    # x_train_img= training_data_orgin[0]
    # R_c1 = x_train_img[:1024]
    # G_c1= x_train_img[1024:2048]
    # B_c1= x_train_img[2048:]
    # xx=np.column_stack((R_c1,G_c1,B_c1))
    # x_train_img = xx.reshape((32,32,3))
    #
    # for i in range(1,len(training_data_orgin)):
    #
    #     R_c = training_data_orgin[i][:1024]
    #     G_c = training_data_orgin[i][1024:2048]
    #     B_c= training_data_orgin[i][2048:]
    #     i_new=np.column_stack((R_c,G_c,B_c))
    #     i_s = i_new.reshape(32,32,3)
    #     x_train_img=np.vstack((x_train_img,i_s))

    training01_label= data01[b'labels']
    training02_label = data02[b'labels']
    training03_label = data03[b'labels']
    training04_label = data04[b'labels']
    training05_label = data05[b'labels']
    training_labels = training01_label+ training02_label+ training03_label+ training04_label + training05_label
    training_labels=np.array(training_labels).reshape(50000,1)
    from keras.utils import  np_utils
    y_label_training_OneHot = np_utils.to_categorical(training_labels)

    training_data_R=training_data_orgin[:,:1024].reshape(50000*1024,)
    training_data_G=training_data_orgin[:,1024:2048].reshape(50000*1024,)
    training_data_B=training_data_orgin[:,2048:].reshape(50000*1024)
    training_X = np.column_stack((training_data_R,training_data_G,training_data_B))
    training_X=training_X.reshape(50000,32,32,3)
    #对x_image进行归一处理
    X_img_train_normalize = training_X.astype('float32')/255.0

    testing_datas = unpickle(os.path.join(datasets,'test_batch'))
    testing_data = testing_datas[b'data']
    testing_data_r=testing_data[:,:1024].reshape(10000*1024,)
    testing_data_g=testing_data[:,1024:2048].reshape(10000*1024,)
    testing_data_b= testing_data[:,2048:].reshape(10000*1024,)
    testing_data_X=np.column_stack((testing_data_r,testing_data_g,testing_data_b))
    testing_data_X=testing_data_X.reshape(10000,32,32,3)

    X_img_test_normalize = testing_data_X.astype('float32')/255.0

    testing_labels=np.array( testing_datas[b'labels']).reshape(10000,1)
    from keras.utils import np_utils
    y_image_testing_OneHot = np_utils.to_categorical(testing_labels)

    import pickle
    cifa10_dict = {'X_train':training_X,'X_test':testing_data_X,'y_train':training_labels,'y_test':testing_labels}
    with open("Cifa10_python",'wb') as f1:
        pickle.dump(cifa10_dict,f1)

