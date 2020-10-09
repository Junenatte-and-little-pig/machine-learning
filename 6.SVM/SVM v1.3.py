# -*- encoding: utf-8 -*-
import pandas as pd
import scipy.io as sio
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def load_data():
    mat = sio.loadmat('./data/ex6data3.mat')
    training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    training['y'] = mat.get('y')

    cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
    cv['y'] = mat.get('yval')
    return training, cv


def main():
    training, cv = load_data()
    # print(training.describe())
    # print(cv.describe())
    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    parameters = {'C': candidate, 'gamma': candidate}
    svc = svm.SVC()
    # GridSearchCV默认会使用一部分数据作为交叉验证集
    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    clf.fit(training[['X1', 'X2']], training['y'])
    print(clf.best_params_)
    print(clf.best_score_)
    ypred = clf.predict(cv[['X1', 'X2']])
    print(metrics.classification_report(cv['y'], ypred))


if __name__ == '__main__':
    main()
