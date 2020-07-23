# -*- encoding: utf-8 -*-
import numpy as np
import scipy.io as sio


def load_date(path, transpose=True):
    data = sio.loadmat(path)
    X = data.get('X')
    y = data.get('y')
    y = y.reshape(y.shape[0])
    if transpose:
        X = np.array([im.reshape([20, 20]).T for im in X])
        X = np.array([im.reshape([400]) for im in X])
    return X, y


def main():
    X, y = load_date('ex3data1.mat')

    # 插入x_0
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    # 在数据集中，因为matlab的索引是从1开始的，所以y的标签是从1到10，并且10所代表的是结果0
    y_matrix = []
    for k in range(1, 11):
        y_matrix.append((y == k).astype(int))
    # 把y == 10移动到最前面
    y_matrix = [y_matrix[-1]] + y_matrix[:-1]
    y = np.array(y_matrix)
    print(X.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
