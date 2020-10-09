# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.io as sio


def load_data():
    mat=sio.loadmat('./data/ex7data1.mat')



def get_X(df):
    ones=pd.DataFrame({'ones':np.ones(len(df))})
    data=pd.concat([ones,df],axis=1)
    return data.iloc[:,:-1].values


def get_y(df):
    return np.array(df.iloc[:,-1])


def normalize_feature(df):
    return df.apply(lambda column:(column-column.mean())/column.std())