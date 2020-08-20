# -*- encoding: utf-8 -*-
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn import svm


def load_data() -> pd.DataFrame:
    mat = sio.loadmat('./data/ex6data1.mat')
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')
    return data


def draw_decision_confidence(data: pd.DataFrame, C=1):
    svc = svm.LinearSVC(C=C, loss='hinge')
    svc.fit(data[['X1', 'X2']], data['y'])
    print(svc.score(data[['X1', 'X2']], data['y']))

    data['SVM{} Confidence'.format(C)] = svc.decision_function(data[['X1', 'X2']])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'],
               cmap='RdBu')
    ax.set_title('SVM (C={}) Decision Confidence'.format(C))
    plt.show()


def main():
    data = load_data()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')
    ax.set_title('Raw data')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.show()

    draw_decision_confidence(data)
    # overfit
    draw_decision_confidence(data, C=100)

    print(data.describe())


if __name__ == '__main__':
    main()
