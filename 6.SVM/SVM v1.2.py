# -*- encoding: utf-8 -*-
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio


def load_data():
    mat=sio.loadmat('./data/ex6data3.mat')
    training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    training['y'] = mat.get('y')

    cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
    cv['y'] = mat.get('yval')
    return training,cv


def main():
    training,cv=load_data()
    print(training.describe())
    print(cv.describe())
    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    combination=[(C,gamma) for C in candidate for gamma in candidate]
    search=[]
    for C,gamma in combination:
        svc=svm.SVC(C=C,gamma=gamma)
        svc.fit(training[['X1','X2']],training['y'])
        search.append(svc.score(training[['X1','X2']],training['y']))
    search=np.array(search)
    best_score=search[np.argmax(search)]
    best_param=combination[np.argmax(search)]
    print(best_score,best_param)


if __name__ == '__main__':
    main()