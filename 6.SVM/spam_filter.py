# -*- encoding: utf-8 -*-
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import scipy.io as sio


def load_data():
    mat_tr=sio.loadmat('./data/spamTrain.mat')
    # print(mat_tr.keys())
    X,y=mat_tr.get('X'),mat_tr.get('y').ravel()
    # print(X.shape,y.shape)
    return X,y
    
    
def load_test():
    mat_test=sio.loadmat('./data/spamTest.mat')
    # print(mat_test.keys())
    X,y=mat_test.get('Xtest'),mat_test.get('ytest').ravel()
    # print(X.shape,y.shape)
    return X,y

def main():
    X,y=load_data()
    Xtest,ytest=load_test()
    svc=svm.SVC()
    svc.fit(X,y)
    pred=svc.predict(Xtest)
    print(metrics.classification_report(ytest,pred))



if __name__ == '__main__':
    main()