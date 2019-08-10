__author__ = "Luke Liu"
#encoding="utf-8"
from  sklearn import  svm
import numpy as np
import  os
from skimage import color
from  skimage.feature import  hog
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__=='__main__':
    dataset_path = '../Cifar10_ConvNetwork-keras/CIFA10/Cifa10_python'

    dataset = unpickle(dataset_path)
    X_image_train = dataset['X_train']
    y_train_label = dataset['y_train']

    X_image_test = dataset["X_test"]
    y_test_label = dataset['y_test']

    label_dict = {0: "airplane", 1: 'automobile', 2: "bird", 3: "cat", 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                  8: 'ship', 9: "truck"}
    import matplotlib.pyplot as  plt
    data_gray = [color.rgb2gray(i) for i in X_image_train]

    ppc = 4
    hog_images = []
    hog_features = []
    for image in data_gray[:1000]:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(ppc, ppc),
        cells_per_block=(2,2), block_norm='L2',visualise=True

)
        hog_images.append(hog_image)
        hog_features.append(fd)
        if len(hog_features)%100==0:
            print("finish {}%".format(len(hog_features)/10.))

    hog_features = np.array(hog_features)
    print(hog_features.shape, len(hog_features))
    train_labels = y_train_label[:1000]
    print(train_labels.shape)
    data_frame = np.hstack((hog_features, train_labels))
    print(data_frame.shape)

    part=800
    tr_data,vali_data = hog_features[:part],hog_features[part:]
    tr_label,vali_label = train_labels[:part].ravel(),train_labels[part:].ravel()
    model=svm.SVC()
    model.fit(tr_data,tr_label)
    pred = model.predict(vali_data)
    from sklearn import metrics
    scores = metrics.accuracy_score(vali_label,pred)
    print("ON testing set",scores)
    scores=metrics.accuracy_score(tr_label,model.predict(tr_data))
    print("On training set",scores)
    # # How to save the trained model and use it into daliy_use
    # from sklearn.externals import joblib
    # joblib.dump(model,"SVM_cifar10.m")
    # # How to use the trained model
    # load_model = joblib.load("SVM_cifar10.m")
