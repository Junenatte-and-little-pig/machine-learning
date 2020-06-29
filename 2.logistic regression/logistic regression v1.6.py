# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# 特征映射
def feature_mapping(x, y, power, as_ndarray=False):
    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
            for i in np.arange(power + 1)
            for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)


def main():
    df = pd.read_csv("ex2data2.txt", names=['test1', 'test2', 'accepted'])

    # describe函数显示数据的情况，包括个数、平均数、标准差、最大最小值等
    # https://www.cnblogs.com/ffli/p/12201448.html
    print("{0:*^50}".format("basic infomation of data"))
    print(df.describe())
    print("{0:*^50}".format(""))

    # 画图查看数据分布
    sns.set(context="notebook", style="ticks",
            font_scale=1.5)
    sns.lmplot('test1', 'test2', hue='accepted', data=df,
               height=6,
               fit_reg=False,
               scatter_kws={"s": 50},
               legend_out=False
               )
    plt.title('Regularized Logistic Regression')
    plt.show()

    x1=np.array(df.test1)
    x2=np.array(df.test2)
    data=feature_mapping(x1,x2,power=6)
    print(data.shape)
    data.describe()


if __name__ == '__main__':
    main()
