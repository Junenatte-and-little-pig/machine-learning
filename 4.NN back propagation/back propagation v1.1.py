# -*- encoding: utf-8 -*-
import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


def cost(theta, X, y):
    m = X.shape[0]

    _, _, _, _, h = feed_forward(theta, X)

    pair_computation = -np.multiply(y, np.log(h)) - np.multiply((1 - y),
                                                                np.log(1 - h))

    return pair_computation.sum() / m


def regularized_cost(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    m = X.shape[0]

    reg_t1 = (l / (2 * m)) * np.power(t1[:, 1:], 2).sum()
    reg_t2 = (l / (2 * m)) * np.power(t2[:, 1:], 2).sum()

    return cost(theta, X, y) + reg_t1 + reg_t2


def gradient(theta, X, y):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    delta1 = np.zeros(t1.shape)
    delta2 = np.zeros(t2.shape)
    a1, z1, a2, z2, h = feed_forward(theta, X)
    for i in range(m):
        a1i = a1[i, :]
        z1i = z1[i, :]
        a2i = a2[i, :]
        hi = h[i, :]
        yi = y[i, :]
        d3i = hi - yi
        z1i = np.insert(z1i, 0, np.ones(1))
        d2i = np.multiply(t2.T @ d3i, sigmoid_gradient(z1i))
        delta2 += d3i.reshape([10, 1]) @ a2i.reshape([1, 26])
        delta1 += d2i[1:].reshape([25, 1]) @ a1i.reshape([1, 401])
    delta1 = delta1 / m
    delta2 = delta2 / m
    return serialize(delta1, delta2)


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)

    t1[:, 0] = 0
    reg_term_d1 = (l / m) * t1
    delta1 = delta1 + reg_term_d1

    t2[:, 0] = 0
    reg_term_d2 = (l / m) * t2
    delta2 = delta2 + reg_term_d2

    return serialize(delta1, delta2)


def expand_array(arr):
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))


def gradient_checking(theta, X, y, epsilon, regularized=False):
    def a_numeric_grad(plus, minus, regularized=False):
        """calculate a partial gradient with respect to 1 theta"""
        if regularized:
            return (regularized_cost(plus, X, y) - regularized_cost(minus, X,
                                                                    y)) / (
                           epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)  # expand to (10285, 10285)
    epsilon_matrix = np.identity(len(theta)) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix

    numeric_grad = np.array(
        [a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
         for i in range(len(theta))])

    analytic_grad = regularized_gradient(theta, X,
                                         y) if regularized else gradient(theta,
                                                                         X, y)

    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(
        numeric_grad + analytic_grad)

    print(
        'If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(
            diff))


def feed_forward(theta, X):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    a1 = X
    z1 = a1 @ t1.T
    a2 = np.insert(sigmoid(z1), 0, np.ones(m), axis=1)
    z2 = a2 @ t2.T
    h = sigmoid(z2)
    return a1, z1, a2, z2, h


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    X = data.get('X')
    y = data.get('y')
    y.reshape(y.shape[0])
    if transpose:
        X = np.array([im.shape((20, 20)).T for im in X])
        X = np.array([im.shape(400) for im in X])
    return X, y


def load_weights(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


def serialize(a: np.ndarray, b: np.ndarray):
    # ravel和flatten都是对数组的扁平化操作
    # ravel原数组和新数组之间，有一个变了另一个也会变
    return np.concatenate((a.flatten(), b.flatten()))


def deserialize(seq):
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


def nn_training(X, y):
    init_theta = random_init(10285)
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res


def show_accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


def main():
    X_raw, y_raw = load_data('ex4data1.mat', transpose=False)
    X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y_raw)
    print(y_onehot)

    t1, t2 = load_weights('ex4weights.mat')
    print(t1.shape, t2.shape)

    theta = serialize(t1, t2)
    # print(theta.shape)

    d1, d2 = deserialize(gradient(theta, X, y_onehot))
    # print(d1.shape, d2.shape)
    # gradient_checking(theta, X, y_onehot, epsilon=0.0001,
    #                   regularized=True)

    res = nn_training(X, y_onehot)
    print(res)

    show_accuracy(res.x, X, y_onehot[:20])


if __name__ == '__main__':
    main()
