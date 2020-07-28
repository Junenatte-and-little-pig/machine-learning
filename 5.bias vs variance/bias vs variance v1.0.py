# -*- encoding: utf-8 -*-
import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from matplotlib import pyplot as plt


def load_data():
    d = sio.loadmat('ex5data1.mat')
    return d['X'].flatten(), d['y'].flatten(), d['Xval'].flatten(), d[
        'yval'].flatten(), d['Xtest'].flatten(), d['ytest'].flatten()


def cost(theta, X, y):
    m = X.shape[0]
    inner = X @ theta - y
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost


def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m


def regularized_cost(theta, X, y, lr=1):
    m = X.shape[0]
    regularized_term = np.power(theta[1:], 2).sum() / (2 * m)
    return cost(theta, X, y) + regularized_term


def regularized_gradient(theta, X, y, lr=1):
    m = X.shape[0]
    regularized_term = theta[1:] / m
    return gradient(theta, X, y) + regularized_term


def linear_regression(X, y, lr=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, lr),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res


def learning_curve(X, y, Xval, yval, lr=1):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m + 1):
        res = linear_regression(X[:i, :], y[:i], lr)
        tc = regularized_cost(res.x, X[:i, :], y[:i], lr)
        cv = regularized_cost(res.x, Xval, yval, lr)
        training_cost.append(tc)
        cv_cost.append(cv)
    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.show()


def main():
    X, y, Xval, yval, Xtest, ytest = load_data()
    X, Xval, Xtest = [
        np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for
        x in (X, Xval, Xtest)]
    res = linear_regression(X, y).get('x')
    b = res[0]
    k = res[1]
    plt.scatter(X[:, 1], y, label="Training data")
    plt.plot(X[:, 1], X[:, 1] * k + b, label="Prediction")
    plt.legend(loc=2)
    plt.show()
    # underfit
    learning_curve(X, y, Xval, yval, lr=0)


if __name__ == '__main__':
    main()
