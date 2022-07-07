import numpy as np
import scipy.io as sio
import csv

import pandas as pd 

# 读取数据
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

    return date_list, susceptible_list, infective_list, recovery_list, death_list

# 读取数据
def load_data2(filename, N=1000000):
    df = pd.read_csv(filename)
    # date_list = np.array(df['date'])
    date_list = [i for i in range(len(df))]

    df['pops'] = 0
    for i in df.index:
        df['pops'][i] = N - df['cum_confirmed'][i] - df['recovered'][i] - df['death'][i]

    df['daily_cases'] = df['cum_confirmed'] - df['cum_confirmed'].shift(1)
    df['daily_cases'][0] = df['cum_confirmed'][0]
    # df['daily_cases'].astype('int')

    df['daily_recovry'] = df['recovered'] - df['recovered'].shift(1)
    df['daily_recovry'][0] = df['recovered'][0]
    # df['daily_recovry'].astype('int')

    df['daily_death'] = df['death'] - df['death'].shift(1)
    df['daily_death'][0] = df['death'][0]
    # df['daily_death'].astype('int')

    susceptible_list = np.array(df['pops'])
    infective_list = np.array(df['daily_cases'])
    recovery_list = np.array(df['daily_recovry'])
    death_list = np.array(df['daily_death'])

    return date_list, susceptible_list, infective_list, recovery_list, death_list

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

def split_data2(date_list, susceptible_list, infective_list, recovery_list, death_list, train_size=0.75):
    train_size = int(len(date_list) * train_size)

    date_train = date_list[0:train_size]
    susceptible_train = susceptible_list[0:train_size]
    infective_train = infective_list[0: train_size]
    recovery_train = recovery_list[0:train_size]
    death_train = death_list[0:train_size]

    date_test = date_list[train_size:-1]
    susceptible_test = susceptible_list[train_size:-1]
    infective_test = infective_list[train_size:-1]
    recovery_test = recovery_list[train_size:-1]
    death_test = death_list[train_size:-1]

    train_data = [date_train, susceptible_train, infective_train, recovery_train, death_train]
    test_data = [date_test, susceptible_test, infective_test, recovery_test, death_test]

    return train_data, test_data

# 拆分数据集，预测后7天


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

# 将数据集拆分为训练集合测试集
def split_3csvData2train_test(date_data, data1, data2, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, date2test, data1_test, data2_test


def load_2csvData(datafile=None):
    csvdata_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata_list.append(int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata = np.array(csvdata_list)
    return csvdate, csvdata


def load_2csvData_cal_S(datafile=None, total_population=100000):
    csvdata2I_list = []
    csvdata2S_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata2I_list.append(int(dataItem2csv[1]))
            csvdata2S_list.append(int(total_population)-int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata2I = np.array(csvdata2I_list)
    csvdata2S = np.array(csvdata2S_list)
    return csvdate, csvdata2I, csvdata2S


def load_3csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    return csvdate, csvdata1, csvdata2


def load_3csvData_cal_S(datafile=None, total_population=100000):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(total_population)-int(dataItem2csv[2]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata2_list)
    return csvdate, csvdata1, csvdata2, csvdata3


def load_4csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(dataItem2csv[3]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata3_list)
    return csvdate, csvdata1, csvdata2, csvdata3


def load_5csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdata4_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(dataItem2csv[3]))
            csvdata4_list.append(int(dataItem2csv[4]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata3_list)
    csvdata4 = np.array(csvdata4_list)
    return csvdate, csvdata1, csvdata2, csvdata3, csvdata4


# 将数据集拆分为训练集合测试集
def split_2csvData2train_test(date_data, data, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data2train = data[0:size2train]/float(normalFactor)

    date2test = date_data[size2train:-1]
    data2test = data[size2train:-1]/float(normalFactor)
    return date2train, data2train, date2test, data2test


# 将数据集拆分为训练集合测试集
def split_3csvData2train_test(date_data, data1, data2, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, date2test, data1_test, data2_test


# 将数据集拆分为训练集合测试集
def split_4csvData2train_test(date_data, data1, data2, data3, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)
    data3_train = data3[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    data3_test = data3[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, data3_train, date2test, data1_test, data2_test, data3_test


# 将数据集拆分为训练集合测试集
def split_5csvData2train_test(date_data, data1, data2, data3, data4, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)
    data3_train = data3[0:size2train] / float(normalFactor)
    data4_train = data4[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    data3_test = data3[size2train:-1] / float(normalFactor)
    data4_test = data4[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, data3_train, data4_train, date2test, data1_test, data2_test, data3_test, data4_test


def randSample_existData(data1, data2, batchsize=1):
    data1_temp = []
    data2_temp = []
    data_length = len(data1)
    indexes = np.random.randint(data_length, size=batchsize)
    for i_index in indexes:
        data1_temp .append(data1[i_index])
        data2_temp .append(data2[i_index])
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    return data1_samples, data2_samples


def randSample_3existData(data1, data2, data3, batchsize=1):
    data1_temp = []
    data2_temp = []
    data3_temp = []
    data_length = len(data1)
    indexes = np.random.randint(data_length, size=batchsize)
    for i_index in indexes:
        data1_temp .append(data1[i_index])
        data2_temp .append(data2[i_index])
        data3_temp.append(data3[i_index])
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)
    data3_samples = np.array(data3_temp)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    data3_samples = data3_samples.reshape(batchsize, 1)
    return data1_samples, data2_samples, data3_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_existData(date_data, data2, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data_temp .append(float(data2[i_index])/float(normalFactor))
    date_samples = np.array(date_temp)
    data_samples = np.array(data_temp)
    date_samples = date_samples.reshape(batchsize, 1)
    data_samples = data_samples.reshape(batchsize, 1)
    return date_samples, data_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_3existData(date_data, data1, data2, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data1_temp = []
    data2_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data1_temp.append(float(data1[i_index]) / float(normalFactor))
        data2_temp .append(float(data2[i_index])/float(normalFactor))

    date_samples = np.array(date_temp)
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)

    date_samples = date_samples.reshape(batchsize, 1)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    return date_samples, data1_samples, data2_samples

def randSample_Normalize_3existData_2(date_data, data1, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data1_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data1_temp.append(float(data1[i_index]) / float(normalFactor))

    date_samples = np.array(date_temp)
    data1_samples = np.array(data1_temp)

    date_samples = date_samples.reshape(batchsize, 1)
    data1_samples = data1_samples.reshape(batchsize, 1)
    return date_samples, data1_samples 


# 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
def sample_testDays_serially(test_date, batch_size):
    day_it = test_date[0:batch_size]
    day_it = np.reshape(day_it, newshape=(batch_size, 1))
    return day_it


# 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
def sample_testData_serially(test_data, batch_size, normalFactor=1000):
    data_it = test_data[0:batch_size]
    data_it = data_it.astype(np.float32)
    data_it = np.reshape(data_it, newshape=(batch_size, 1))
    data_it = data_it/float(normalFactor)
    return data_it

def sample_testData_serially2(test_data, batch_size):
    data_it = test_data[0:batch_size]
    data_it = data_it.astype(np.float32)
    data_it = np.reshape(data_it, newshape=(batch_size, 1))
    return data_it


# ---------------------------------------------- 数据集的生成 ---------------------------------------------------
#  方形区域[a,b]^n生成随机数, n代表变量个数
def rand_it(batch_size, variable_dim, region_a, region_b):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    x_it = x_it.astype(np.float32)
    return x_it


def rand_bd_1D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 1:
        x_left_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_a
        x_right_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_b
        return x_left_bd, x_right_bd
    else:
        return


def rand_bd_2D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 2:
        x_left_bd = (region_b-region_a) * np.random.random([batch_size, 2]) + region_a   # 浮点数都是从0-1中随机。
        for ii in range(batch_size):
            x_left_bd[ii, 0] = region_a

        x_right_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            x_right_bd[ii, 0] = region_b

        y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            y_bottom_bd[ii, 1] = region_a

        y_top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            y_top_bd[ii, 1] = region_b

        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
        return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd
    else:
        return


def rand_bd_3D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 3:
        bottom_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            bottom_bd[ii, 2] = region_a

        top_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            top_bd[ii, 2] = region_b

        left_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            left_bd[ii, 1] = region_a

        right_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            right_bd[ii, 1] = region_b

        front_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            front_bd[ii, 0] = region_b

        behind_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            behind_bd[ii, 0] = region_a

        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)
        front_bd = front_bd.astype(np.float32)
        behind_bd = behind_bd.astype(np.float32)
        return bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd
    else:
        return


def rand_bd_4D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    variable_dim = int(variable_dim)

    x0a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x0a[ii, 0] = region_a

    x0b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x0b[ii, 0] = region_b

    x1a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x1a[ii, 1] = region_a

    x1b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x1b[ii, 1] = region_b

    x2a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x2a[ii, 2] = region_a

    x2b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x2b[ii, 2] = region_b

    x3a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x3a[ii, 3] = region_a

    x3b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x3b[ii, 3] = region_b

    x0a = x0a.astype(np.float32)
    x0b = x0b.astype(np.float32)

    x1a = x1a.astype(np.float32)
    x1b = x1b.astype(np.float32)

    x2a = x2a.astype(np.float32)
    x2b = x2b.astype(np.float32)

    x3a = x3a.astype(np.float32)
    x3b = x3b.astype(np.float32)

    return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b


def rand_bd_5D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 5:
        x0a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x0a[ii, 0] = region_a

        x0b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x0b[ii, 0] = region_b

        x1a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x1a[ii, 1] = region_a

        x1b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x1b[ii, 1] = region_b

        x2a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x2a[ii, 2] = region_a

        x2b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x2b[ii, 2] = region_b

        x3a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x3a[ii, 3] = region_a

        x3b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x3b[ii, 3] = region_b

        x4a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x4a[ii, 4] = region_a

        x4b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x4b[ii, 4] = region_b

        x0a = x0a.astype(np.float32)
        x0b = x0b.astype(np.float32)

        x1a = x1a.astype(np.float32)
        x1b = x1b.astype(np.float32)

        x2a = x2a.astype(np.float32)
        x2b = x2b.astype(np.float32)

        x3a = x3a.astype(np.float32)
        x3b = x3b.astype(np.float32)

        x4a = x4a.astype(np.float32)
        x4b = x4b.astype(np.float32)
        return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, x4a, x4b
    else:
        return
