# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from sklearn.metrics import classification_report


# 读取特征
def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    # 按列合并
    data = pd.concat([ones, df], axis=1)
    # 返回ndarray
    # as_matrix方法弃用
    # values属性不被推荐
    return data.iloc[:, :-1].to_numpy()


# 读取标签
def get_Y(df):
    return np.array(df.iloc[:, -1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, Y):
    return np.mean(
        -Y * np.log(sigmoid(X @ theta)) - (1 - Y) * np.log(
            1 - sigmoid(X @ theta)))


def gradient(theta, X, Y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - Y)


def predict(X, theta):
    prob = sigmoid(X @ theta)
    # astype修改数据类型
    return (prob >= 0.5).astype(int)


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
               fit_reg=False, scatter_kws={"s": 50}, legend_out=False)
    plt.show()

    # 查看数据维度
    print("{0:*^50}".format("dimension of data"))
    X = get_X(data)
    print('X: ', X.shape, type(X))
    Y = get_Y(data)
    print('Y: ', Y.shape, type(Y))
    print("{0:*^50}".format(''))

    # 拟合参数
    print("{0:*^50}".format("parameters"))
    theta = np.zeros(3)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    res = opt.minimize(fun=cost, x0=theta, args=(X, Y), method="Newton-CG",
                       jac=gradient)
    print(res)
    print("{0:*^50}".format(""))

    # 数据集预测和评价
    print("{0:*^50}".format("predict & evaluation on dataset"))
    final_theta = res.x
    y_pred = predict(X, final_theta)
    print(classification_report(Y, y_pred))
    print("{0:*^50}".format(""))

    coef = -(res.x / res.x[2])  # find the equation
    x = np.arange(130, step=0.1)
    y = coef[0] + coef[1] * x
    sns.set(context="notebook", style="ticks", font_scale=1.5)

    sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
               height=6,
               fit_reg=False,
               scatter_kws={"s": 25}
               )

    plt.plot(x, y, 'grey')
    plt.xlim(0, 130)
    plt.ylim(0, 130)
    plt.title('Decision Boundary')
    plt.show()


if __name__ == '__main__':
    main()
