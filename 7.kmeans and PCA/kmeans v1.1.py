# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sns
from sklearn.cluster import KMeans


def load_data(path):
    mat = sio.loadmat(path)
    # print(mat.keys())
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    # print(data.head())
    return data


def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c


def main():
    data = load_data('./data/ex7data2.mat')
    sk_kmeans = KMeans(n_clusters=3)
    sk_kmeans.fit(data)
    sk_C = sk_kmeans.predict(data)
    data_with_C = combine_data_C(data, sk_C)
    sns.lmplot('X1', 'X2', hue='C', data=data_with_C, fit_reg=False)
    plt.show()


if __name__ == '__main__':
    main()
