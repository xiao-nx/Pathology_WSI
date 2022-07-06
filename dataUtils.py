import os
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import time
import DNN_tools

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


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of dealing with SIR: %s\n' % str(R_dic['model2SIRD']), log_fileout)
    DNN_tools.log_string('Network model of dealing with parameters: %s\n' % str(R_dic['model2paras']), log_fileout)
    if str.upper(R_dic['model2SIRD']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for SIRD: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for SIRD: %s\n' % str(R_dic['actIn_Name2SIRD']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for SIRD: %s\n' % str(R_dic['act_Name2SIRD']), log_fileout)

    if str.upper(R_dic['model2paras']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for parameter: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for parameter: %s\n' % str(R_dic['actIn_Name2paras']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for parameter: %s\n' % str(R_dic['act_Name2paras']), log_fileout)

    DNN_tools.log_string('hidden layers for SIR: %s\n' % str(R_dic['hidden2SIRD']), log_fileout)
    DNN_tools.log_string('hidden layers for parameters: %s\n' % str(R_dic['hidden2para']), log_fileout)

    if str.upper(R_dic['model2SIRD']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2SIRD']), log_fileout)
        DNN_tools.log_string('Repeat the high-frequency scale or not for SIR-NN: %s\n' % str(R_dic['if_repeat_High_freq2SIRD']), log_fileout)
    if str.upper(R_dic['model2paras']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2paras']), log_fileout)
        DNN_tools.log_string('Repeat the high-frequency scale or not for para-NN: %s\n' % str(R_dic['if_repeat_High_freq2paras']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']), log_fileout)

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']), log_fileout)

    # DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    # DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    # DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)

# 记录字典中的一些设置
def dictionary_out2file2(R_dic, log_fileout):
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of dealing with SIR: %s\n' % str(R_dic['sird_network']), log_fileout)
    # DNN_tools.log_string('Network model of dealing with parameters: %s\n' % str(R_dic['model2paras']), log_fileout)
    if str.upper(R_dic['sird_network']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for SIRD: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for SIRD: %s\n' % str(R_dic['activateIn_sird']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for SIRD: %s\n' % str(R_dic['activate_sird']), log_fileout)

    if str.upper(R_dic['params_network']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for parameter: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for parameter: %s\n' % str(R_dic['activateIn_params']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for parameter: %s\n' % str(R_dic['activate_params']), log_fileout)

    DNN_tools.log_string('hidden layers for SIR: %s\n' % str(R_dic['hidden_sird']), log_fileout)
    DNN_tools.log_string('hidden layers for parameters: %s\n' % str(R_dic['hidden_params']), log_fileout)

    if str.upper(R_dic['sird_network']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2SIRD']), log_fileout)
        # DNN_tools.log_string('Repeat the high-frequency scale or not for SIR-NN: %s\n' % str(R_dic['if_repeat_High_freq2SIRD']), log_fileout)
    if str.upper(R_dic['params_network']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2paras']), log_fileout)
        # DNN_tools.log_string('Repeat the high-frequency scale or not for para-NN: %s\n' % str(R_dic['if_repeat_High_freq2paras']), log_fileout)

    # DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    # DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    # DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)

    if R_dic['optimizer'] == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer'], R_dic['momentum']), log_fileout)

    # if R_dic['activate_stop'] != 0:
    #     DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['train_epoches']), log_fileout)
    # else:
    #     DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['train_epoches']), log_fileout)

    DNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']), log_fileout)

    # DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']), log_fileout)

    # DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    # DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    # DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


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