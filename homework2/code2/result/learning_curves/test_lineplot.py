#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 2021/10/27 15:38
# @Author : Jin Xuefeng
# @File   : test_lineplot.py
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.set()


def smooth(data, wd=2):
    """
    :param data: ndarray，一维或二维
    :param wd:
    :return:
    """
    if not (isinstance(wd, int) and wd > 0):
        raise ValueError('wd must be a positive integer')
    elif 1 == wd:
        return data
    else:
        weight = np.ones(wd) / wd
        if 1 == data.ndim:
            return np.convolve(weight, data, "valid")
        elif 2 == data.ndim:
            smooth_data = []
            for d in data:
                d = np.convolve(weight, d, "valid")
                smooth_data.append(d)
            return np.array(smooth_data)
        else:
            raise ValueError('data must be a one-dimensional or two-dimensional ndarray')


def PLOT_REINFOTRCE():
    with open('../train/pg1_seed1.pkl', 'rb') as fp:
        r1_1 = pickle.load(fp)
    with open('../train/pg1_seed2.pkl', 'rb') as fp:
        r1_2 = pickle.load(fp)
    with open('../train/pg1_seed3.pkl', 'rb') as fp:
        r1_3 = pickle.load(fp)
    with open('../train/pg1_seed4.pkl', 'rb') as fp:
        r1_4 = pickle.load(fp)
    r1 = np.array([r1_1, r1_2, r1_3, r1_4])

    with open('../train/pg2_seed1.pkl', 'rb') as fp:
        r2_1 = pickle.load(fp)
    with open('../train/pg2_seed2.pkl', 'rb') as fp:
        r2_2 = pickle.load(fp)
    with open('../train/pg2_seed3.pkl', 'rb') as fp:
        r2_3 = pickle.load(fp)
    with open('../train/pg2_seed4.pkl', 'rb') as fp:
        r2_4 = pickle.load(fp)
    r2 = np.array([r2_1, r2_2, r2_3, r2_4])

    wd = 10
    r1 = smooth(r1, wd)
    r2 = smooth(r2, wd)


    data = [r1, r2]
    label = ['REINFORCE', 'REINFORCE with baseline']
    df=[]
    ax = range(1000-wd+1)   # x轴刻度
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i], columns=ax).melt(var_name='episode',value_name='return'))
        df[i]['algo'] = label[i]
    df=pd.concat(df, ignore_index=True)
    # print(df)
    sns.lineplot(x="episode", y="return", hue="algo", data=df)
    # sns.color_palette("bright")
    plt.legend(loc='upper left')
    plt.title("")

    # plt.savefig(fname='./png/'+'f1.png')
    plt.show()


def PLOT_TASK2():
    with open('../train/pg1_seed1.pkl', 'rb') as fp:
        r1_1 = pickle.load(fp)
    with open('../train/pg1_seed2.pkl', 'rb') as fp:
        r1_2 = pickle.load(fp)
    with open('../train/pg1_seed3.pkl', 'rb') as fp:
        r1_3 = pickle.load(fp)
    with open('../train/pg1_seed4.pkl', 'rb') as fp:
        r1_4 = pickle.load(fp)
    r1 = np.array([r1_1, r1_2, r1_3, r1_4])

    with open('../train/pg2_seed1.pkl', 'rb') as fp:
        r2_1 = pickle.load(fp)
    with open('../train/pg2_seed2.pkl', 'rb') as fp:
        r2_2 = pickle.load(fp)
    with open('../train/pg2_seed3.pkl', 'rb') as fp:
        r2_3 = pickle.load(fp)
    with open('../train/pg2_seed4.pkl', 'rb') as fp:
        r2_4 = pickle.load(fp)
    r2 = np.array([r2_1, r2_2, r2_3, r2_4])

    wd = 10
    r1 = smooth(r1, wd)
    r2 = smooth(r2, wd)

    data = [r1, r2]
    label = ['REINFORCE', 'REINFORCE with baseline']
    df = []
    ax = range(1000 - wd + 1)  # x轴刻度
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i], columns=ax).melt(var_name='episode', value_name='return'))
        df[i]['algo'] = label[i]
    df = pd.concat(df, ignore_index=True)
    # print(df)
    sns.lineplot(x="episode", y="return", hue="algo", data=df)
    # sns.color_palette("bright")
    plt.legend(loc='upper left')
    plt.title("")

    # plt.savefig(fname='./png/' + 'f1.png')
    plt.show()


if __name__ == '__main__':
    PLOT_REINFOTRCE()