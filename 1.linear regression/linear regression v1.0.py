# -*- encoding: utf-8 -*-
# numpy 1.16.1
import pandas as pd


def main():
    # 读取数据并赋予列名
    # https://blog.csdn.net/zjyklwg/article/details/79556545
    # 返回值为DataFrame结构的数据，形象的看为一个二维表结构
    # 行为索引，列为标签
    # https://www.jianshu.com/p/2ef4f057fe0d
    df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

    # 展示数据基本信息，包含索引范围、各个列非空数据个数、数据类型、占用空间大小
    print("{0:*^50}".format("type of df"))
    df.info()
    print("{0:*^50}".format(''))

    # 展示数据表前5行数据
    print("{0:*^50}".format("first 5 lines of df"))
    print(df.head())
    print("{0:*^50}".format(''))


if __name__ == '__main__':
    main()
