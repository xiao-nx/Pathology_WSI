import os
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import time

def load_data(fileName, N=3458790):
    '''
    fileName: path of csv file
    N: total population in the city(country)
    '''
    df = pd.read_csv(fileName)
    # date_list = np.array(df['date'])
    date_list = [i for i in range(len(df))]
    infective_list = np.array(df['cum_confirmed'])
    recovery_list = np.array(df['recovered'])
    death_list = np.array(df['death'])
    df['pops'] = 0
    for i in df.index:
        df['pops'][i] = N - df['cum_confirmed'][i] - df['recovered'][i] - df['death'][i]

    susceptible_list = np.array(df['pops'])

    data_list = [date_list, susceptible_list, infective_list, recovery_list, death_list]

    return data_list

# 拆分数据为训练集和测试集
def split_data(date_list, susceptible_list, infective_list, recovery_list, death_list, train_size=0.75, normalFactor=1.0):
    train_size = int(len(date_list) * train_size)
    # test = len(date_list) - train_size

    date_train = date_list[0:train_size]
    susceptible_train = susceptible_list[0:train_size] /float(normalFactor)
    infective_train = infective_list[0: train_size] /float(normalFactor)
    recovery_train = recovery_list[0:train_size] /float(normalFactor)
    death_train = death_list[0:train_size] /float(normalFactor)

    date_test = date_list[train_size:-1]
    susceptible_test = susceptible_list[train_size:-1] /float(normalFactor)
    infective_test = infective_list[train_size:-1] /float(normalFactor)
    recovery_test = recovery_list[train_size:-1] /float(normalFactor)
    death_test = death_list[train_size:-1] /float(normalFactor)

    train_data = [date_train, susceptible_train, infective_train, recovery_train, death_train]
    test_data = [date_test, susceptible_test, infective_test, recovery_test, death_test]

    return train_data, test_data

# 从总体数据集中载入部分数据作为训练集
# 窗口滑动采样时，batch size = window_size 
def sample_data(date_data, s_data, i_data,r_data, d_data, window_size=1, sampling_opt=None):
    date_temp = list()
    s_temp = list()
    i_temp = list()
    r_temp = list()
    d_temp = list()
    data_length = len(date_data)
    if sampling_opt.lower() == 'random_sample':
        indexes = np.random.randint(data_length, size=window_size)
    elif sampling_opt.lower() == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=window_size)
        indexes = np.sort(indexes_temp)
    elif sampling_opt.lower() == 'sequential_sort':
        index_base = np.random.randint(data_length - window_size, size=1)
        indexes = np.arange(index_base, index_base + window_size)
    for i_index in indexes:
        date_temp.append(float(date_data[i_index]))
        # data_temp.append(float(data2[i_index])/float(normalFactor))
        s_temp.append(float(s_data[i_index]))
        i_temp.append(float(i_data[i_index]))
        r_temp.append(float(r_data[i_index]))
        d_temp.append(float(d_data[i_index]))

    date_samples = np.array(date_temp)
    # data_samples = np.array(data_temp)
    s_samples = np.array(s_temp)
    i_samples = np.array(i_temp)
    r_samples = np.array(r_temp)
    d_samples = np.array(d_temp)
    date_samples = date_samples.reshape(window_size, 1)
    # data_samples = data_samples.reshape(batchsize, 1)
    s_samples = s_samples.reshape(window_size, 1)
    i_samples = i_samples.reshape(window_size, 1)
    r_samples = r_samples.reshape(window_size, 1)
    d_samples = d_samples.reshape(window_size, 1)

    return date_samples, s_samples, i_samples, r_samples, d_samples

def compute_mse_res(data_obs, nn_predict):
    point_ERR2I = np.square(nn_predict - data_obs)
    mse = np.mean(point_ERR2I)
    res = mse / np.mean(np.square(data_obs))

    return mse, res

# point_ERR2I = np.square(i_nn2test - i_obs_test)
# test_mse2I = np.mean(point_ERR2I)
# test_mse2I_all.append(test_mse2I)
# test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
# test_rel2I_all.append(test_rel2I)