# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
from skimage import io as iio
from sklearn.cluster import KMeans


def load_data():
    pic = iio.imread('./data/bird_small.png') / 255.
    iio.imshow(pic)
    plt.show()
    # print(pic.shape)
    data = pic.reshape(128 * 128, 3)
    # print(data.shape)
    return data


def main():
    data = load_data()
    kmeans = KMeans(n_clusters=16, n_init=100)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    print(centroids)
    C = kmeans.predict(data)
    print(C)
    print(centroids[C])
    compressed_pic = centroids[C].reshape((128, 128, 3))
    iio.imshow(compressed_pic)
    plt.show()


if __name__ == '__main__':
    main()
