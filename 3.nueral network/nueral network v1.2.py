# -*- encoding: utf-8 -*-
import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_date(path, transpose=True):
    data = sio.loadmat(path)
    X = data.get('X')
    y = data.get('y')
    y = y.reshape(y.shape[0])
    if transpose:
        X = np.array([im.reshape([20, 20]).T for im in X])
        X = np.array([im.reshape([400]) for im in X])
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(
        1 - sigmoid(X @ theta)))


def regularized_cost(theta, X, y, l=1):
    theta_j1_to_jn = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_jn, 2).sum()
    return cost(theta, X, y) + regularized_term


def gradient(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def regularized_gradient(theta, X, y, l=1):
    theta_j1_to_jn = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_jn
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient(theta, X, y) + regularized_term


def logistic_regression(X, y, l=1):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    final_theta = res.x
    return final_theta


def predict(X, theta):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)


def main():
    X, raw_y = load_date('ex3data1.mat')

    # 插入x_0
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    # 在数据集中，因为matlab的索引是从1开始的，所以y的标签是从1到10，并且10所代表的是结果0
    y_matrix = []
    for k in range(1, 11):
        y_matrix.append((raw_y == k).astype(int))
    # 把y == 10移动到最前面
    y_matrix = [y_matrix[-1]] + y_matrix[:-1]
    y = np.array(y_matrix)
    print(X.shape)
    print(y.shape)

    k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])
    k_prob = sigmoid(X @ k_theta.T)
    print(k_prob)
    y_pred = np.argmax(k_prob, axis=1)
    print(y_pred)

    y_answer=raw_y.copy()
    y_answer[y_answer == 10] = 0
    print(classification_report(y_answer, y_pred))


if __name__ == '__main__':
    main()
