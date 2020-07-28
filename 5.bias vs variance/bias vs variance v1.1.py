# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
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


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()
    # 保证维度相同又不惩罚theta_0
    regularized_term[0] = 0
    regularized_term = (l / m) * regularized_term
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
    return np.argmin(cv_cost)


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    return df.to_numpy() if as_ndarray else df


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def prepare_poly_data(*args, power):
    def prepare(x):
        df = poly_features(x, power=power)
        ndarr = normalize_feature(df).to_numpy()
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]


def main():
    X, y, Xval, yval, Xtest, ytest = load_data()
    X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
    # outfit
    # learning_curve(X_poly, y, Xval_poly, yval, lr=0)
    # learning_curve(X_poly, y, Xval_poly, yval, lr=1)
    # learning_curve(X_poly, y, Xval_poly, yval, lr=100)
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    for lr in l_candidate:
        theta = linear_regression(X_poly, y, lr).success
        print('test cost(l={}) = {}'.format(lr, cost(theta, Xtest_poly, ytest)))


if __name__ == '__main__':
    main()
