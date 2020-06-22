# -*- encoding: utf-8 -*-
import pandas as pd


def main():
    data = pd.read_csv("ex2data1.txt", names=['exam1', 'exam2', 'admitted'])

    # describe函数显示数据的情况，包括个数、平均数、标准差、最大最小值等
    # https://www.cnblogs.com/ffli/p/12201448.html
    print("{0:*^50}".format("basic infomation of data"))
    print(data.describe())
    print("{0:*^50}".format(""))


if __name__ == '__main__':
    main()
