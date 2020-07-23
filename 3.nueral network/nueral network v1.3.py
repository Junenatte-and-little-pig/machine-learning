# -*- encoding: utf-8 -*-
import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_date(path, transpose=True):
    data = sio.loadmat(path)
    X = data.get('X')
    y = data.get('y')
    y = y.reshape(y.shape[0])
    if transpose:
        X = np.array([im.reshape([20, 20]).T for im in X])
        X = np.array([im.reshape([400]) for im in X])
    return X, y


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


def main():
    theta1, theta2 = load_weight('ex3weights.mat')
    print(theta1.shape, theta2.shape)

    X, y = load_date('ex3data1.mat',transpose=False)

    # 插入x_0
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    a1 = X
    z2 = a1 @ theta1.T
    print(z2.shape)

    z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
    a2 = sigmoid(z2)
    print(a2.shape)

    z3 = a2 @ theta2.T
    print(z3.shape)

    a3 = sigmoid(z3)
    print(a3)

    y_pred = np.argmax(a3, axis=1) + 1
    print(y_pred)

    print(classification_report(y, y_pred))


if __name__ == '__main__':
    main()
