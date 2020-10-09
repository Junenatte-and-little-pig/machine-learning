# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns


def load_data(path):
    mat = sio.loadmat(path)
    # print(mat.keys())
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    # print(data.head())
    return data


def random_init(data, k):
    return data.sample(k).values


def _find_your_cluster(x, centroids):
    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1,
                                    arr=centroids - x)
    return np.argmin(distances)


def assign_cluster(data, centroids):
    return np.apply_along_axis(lambda x: _find_your_cluster(x, centroids),
                               axis=1, arr=data.values)


def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c


def new_centroids(data, C):
    data_with_C = combine_data_C(data, C)
    return data_with_C.groupby('C', as_index=False).mean().sort_values(
        by='C').drop('C', axis=1).values


def cost(data, centroids, C):
    expend_C_with_centroids = centroids[C]
    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1,
                                    arr=data.values - expend_C_with_centroids)
    return distances.mean()


def _kmeans_iter(data, k, epoch=100, tol=0.0001):
    centroids = random_init(data, k)
    cost_progress = []
    C = assign_cluster(data, centroids)
    for i in range(epoch):
        print('running epoch:{}'.format(i))
        C = assign_cluster(data, centroids)
        centroids = new_centroids(data, C)
        cost_progress.append(cost(data, centroids, C))
        # sns.lmplot('X1','X2',hue='C',data=combine_data_C(data,C),fit_reg=False)
        # plt.show()
        # 当代价值提升效果小于阈值时，提前结束循环
        if len(cost_progress) > 1:
            if np.abs(cost_progress[-1] - cost_progress[-2]) / cost_progress[
                -1] < tol:
                break
    return C, centroids, cost_progress[-1]


def kmeans(data, k, epoch=100, init=10):
    trials = np.array([_kmeans_iter(data, k, epoch) for _ in range(init)])
    least_cost_idx = np.argmin(trials[:, -1])
    return trials[least_cost_idx]


def main():
    # data1=load_data('./data/ex7data1.mat')
    # sns.lmplot('X1','X2',data=data1,fit_reg=False)
    # plt.show()
    data2 = load_data('./data/ex7data2.mat')
    sns.lmplot('X1', 'X2', data=data2, fit_reg=False)
    plt.show()

    best_C, best_centroids, least_cost = kmeans(data2, 3)
    # print(best_C)
    # print(best_centroids)
    # print(least_cost)
    data_with_c = combine_data_C(data2, best_C)
    sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
    plt.show()


if __name__ == '__main__':
    main()
