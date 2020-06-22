# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    data = pd.read_csv("ex2data1.txt", names=['exam1', 'exam2', 'admitted'])

    # describe函数显示数据的情况，包括个数、平均数、标准差、最大最小值等
    # https://www.cnblogs.com/ffli/p/12201448.html
    print("{0:*^50}".format("basic infomation of data"))
    print(data.describe())
    print("{0:*^50}".format(""))

    # 画图查看数据分布
    sns.set(context="notebook", style="darkgrid",
            palette=sns.color_palette("RdBu", 2))
    sns.lmplot('exam1', 'exam2', hue='admitted', data=data, height=6,
               fit_reg=False, scatter_kws={"s": 50},legend_out=False)
    plt.show()


if __name__ == '__main__':
    main()
