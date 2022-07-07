"""
@author: LXA
Benchmark Code of SIRD model
2022-06-18
"""
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import platform
import shutil
import DNN_base
import DNN_tools
import DNN_data
import plotData
import saveData
import dataUtils
# tf2兼容tf1
tf.compat.v1.disable_eager_execution()

# 在模型中除以N

DECAY_STEPS = 20000
DECAY_RATE = 0.9

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_fname', type=str, default='./data/minnesota3.csv',
                    help='data path')

parser.add_argument('--output_dir', type=str, default='./output_SIRD',
                    help='data path')

parser.add_argument('--train_epoches', type=int, default=200000,
                    help='max epoch during training.')

parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer')

parser.add_argument('--epoches_per_eval', type=int, default=10000,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--eqs_name', type=str, default='SIRD',
                    help='Whether to use debugger to track down bad values during training.')

parser.add_argument('--batach_size', type=int, default=16,
                    help='.')

parser.add_argument('--sample_method', type=str, default='rand_sample_sort',
                    help='. : random_sample, rand_sample_sort, sequential_sort')

parser.add_argument('--sird_network', type=str, default='DNN_FOURIERBASE',
                    help='network archtecture:' 'DNN, DNN_FOURIERBASE, DNN_SCALE')

parser.add_argument('--params_network', type=str, default='DNN_FOURIERBASE',
                    help='network archtecture:' 'DNN, DNN_FOURIERBASE, DNN_SCALE')

parser.add_argument('--hidden_sird', type=set, default=([35, 50, 30, 30, 20]),
                    help='hidden layers:'
                         '(80, 80, 60, 40, 40, 20)'
                         '(100, 100, 80, 60, 60, 40)'
                         '(200, 100, 100, 80, 50, 50)')

parser.add_argument('--hidden_params', type=set, default=([35, 50, 30, 30, 20]),
                    help='hidden layers:'
                         '(80, 80, 60, 40, 40, 20)'
                         '(100, 100, 80, 60, 60, 40)'
                         '(200, 100, 100, 80, 50, 50)')  

parser.add_argument('--loss_function', type=str, default='L2_loss',
                    help='loss function:' 'L2_loss, lncosh_loss')      
# SIRD和参数网络模型的激活函数的选择
parser.add_argument('--activateIn_sird', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')  
parser.add_argument('--activate_sird', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')  
parser.add_argument('--activateIn_params', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')  
parser.add_argument('--activate_params', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')                                     

parser.add_argument('--init_penalty2predict_true', type=int, default=50, # 预测值和真值的误差惩罚因子初值,用于处理具有真实值的变量
                    help='Regularization parameter for boundary conditions.')

parser.add_argument('--activate_stage_penalty', type=bool, default=True,
                    help='Whether to use Regularization parameter for boundary conditions.')

parser.add_argument('--regular_method', type=str, default='L2',
                    help='The method of regular weights and biases:' 'L0, L1')

parser.add_argument('--regular_weight', type=float, default=0.005, # 神经网络参数的惩罚因子
                    help='Regularization parameter for weights.' '0.00001, 0.00005, 0.0001, 0.0005, 0.001')

parser.add_argument('--initial_learning_rate', type=float, default=0.01,
                    help='.'
                    '0.1, 0.01, 0.05, 0,001')

parser.add_argument('--decay_steps', type=float, default=0.01,
                    help='.' '0.1, 0.01, 0.05, 0,001')

parser.add_argument('--population', type=float, default=3450000,
                    help='.')

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

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']), log_fileout)

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']), log_fileout)

    DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_nt, penalty_wb2s, penalty_wb2i, penalty_wb2r,
                        penalty_wb2d, loss_s, loss_i, loss_r, loss_d, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('penalty for difference of predict and true : %f' % temp_penalty_nt)
    print('penalty weights and biases for S: %f' % penalty_wb2s)
    print('penalty weights and biases for I: %f' % penalty_wb2i)
    print('penalty weights and biases for params: %f' % penalty_wb2r)
    print('penalty weights and biases for D: %f' % penalty_wb2d)
    print('loss for S: %.16f' % loss_s)
    print('loss for I: %.16f' % loss_i)
    print('loss for R: %.16f' % loss_r)
    print('loss for D: %.16f' % loss_d)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('penalty for difference of predict and true : %f' % temp_penalty_nt, log_out)
    DNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2s, log_out)
    DNN_tools.log_string('penalty weights and biases for I: %f' % penalty_wb2i, log_out)
    DNN_tools.log_string('penalty weights and biases for params: %.10f' % penalty_wb2r, log_out)
    DNN_tools.log_string('penalty weights and biases for D: %.10f' % penalty_wb2d, log_out)
    DNN_tools.log_string('loss for S: %.16f' % loss_s, log_out)
    DNN_tools.log_string('loss for I: %.16f' % loss_i, log_out)
    DNN_tools.log_string('loss for params: %.16f' % loss_r, log_out)
    DNN_tools.log_string('loss for D: %.16f' % loss_d, log_out)


def solve_SIRD2COVID(params):
    log_out_path = params['FolderName']        # 将路径从字典 params 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(params, log_fileout)

    log2trianSolus = open(os.path.join(log_out_path, 'train_Solus.txt'), 'w')      # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus = open(os.path.join(log_out_path, 'test_Solus.txt'), 'w')        # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus2 = open(os.path.join(log_out_path, 'test_Solus_temp.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    log2testParas = open(os.path.join(log_out_path, 'test_Paras.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    trainSet_szie = params['size2train']                   # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchSize_train = params['batch_size2train']           # 训练批量的大小,该值远小于训练集大小
    batchSize_test = params['batch_size2test']             # 测试批量的大小,该值小于等于测试集大小
    pt_penalty_init = params['init_penalty2predict_true']  # 预测值和真值得误差惩罚因子的初值,用于处理那些具有真实值得变量
    wb_penalty = params['regular_weight']                  # 神经网络参数的惩罚因子
    lr_decay = params['lr_decay']                          # 学习率额衰减
    init_lr = params['learning_rate']                      # 初始学习率

    act_func2SIRD = params['act_Name2SIRD']                 # S, I, params D 四个神经网络的隐藏层激活函数
    act_func2paras = params['act_Name2paras']              # 参数网络的隐藏层激活函数

    input_dim = params['input_dim']                        # 输入维度
    out_dim = params['output_dim']                         # 输出维度

    hidden_sird = params['hidden2SIRD']
    hidden_para = params['hidden2para']

    if str.upper(params['model2SIRD']) == 'DNN_FOURIERBASE':
        Weight2S, Bias2S = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, 'wb_s')
        Weight2I, Bias2I = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, 'wb_i')
        Weight2R, Bias2R = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, 'wb_r')
        Weight2D, Bias2D = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, 'wb_d')
    else:
        Weight2S, Bias2S = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, 'wb_s')
        Weight2I, Bias2I = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, 'wb_i')
        Weight2R, Bias2R = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, 'wb_r')
        Weight2D, Bias2D = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, 'wb_d')

    if str.upper(params['model2paras']) == 'DNN_FOURIERBASE':
        Weight2beta, Bias2beta = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, 'wb_beta')
        Weight2gamma, Bias2gamma = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, 'wb_gamma')
        Weight2mu, Bias2mu = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, 'wb_mu')
    else:
        Weight2beta, Bias2beta = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, 'wb_beta')
        Weight2gamma, Bias2gamma = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, 'wb_gamma')
        Weight2mu, Bias2mu = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, 'wb_mu')

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (params['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            T_it = tf.compat.v1.placeholder(tf.float32, name='T_it', shape=[None, out_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[None, out_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[None, out_dim])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[None, out_dim])
            predict_true_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='pt_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            freq2SIRD = params['freq2SIRD']
            if 'DNN' == str.upper(params['model2SIRD']):
                SNN_temp = DNN_base.DNN(T_it, Weight2S, Bias2S, hidden_sird, activateIn_name=params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                INN_temp = DNN_base.DNN(T_it, Weight2I, Bias2I, hidden_sird, activateIn_name=params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                RNN_temp = DNN_base.DNN(T_it, Weight2R, Bias2R, hidden_sird, activateIn_name=params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                DNN_temp = DNN_base.DNN(T_it, Weight2D, Bias2D, hidden_sird, activateIn_name=params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
            elif 'DNN_SCALE' == str.upper(params['model2SIRD']):
                SNN_temp = DNN_base.DNN_scale(T_it, Weight2S, Bias2S, hidden_sird, freq2SIRD,
                                              activateIn_name=params['actIn_Name2SIR'], activate_name=act_func2SIRD)
                INN_temp = DNN_base.DNN_scale(T_it, Weight2I, Bias2I, hidden_sird, freq2SIRD,
                                              activateIn_name=params['actIn_Name2SIR'], activate_name=act_func2SIRD)
                RNN_temp = DNN_base.DNN_scale(T_it, Weight2R, Bias2R, hidden_sird, freq2SIRD,
                                              activateIn_name=params['actIn_Name2SIR'], activate_name=act_func2SIRD)
                DNN_temp = DNN_base.DNN_scale(T_it, Weight2D, Bias2D, hidden_sird, freq2SIRD,
                                              activateIn_name=params['actIn_Name2SIR'], activate_name=act_func2SIRD)
            elif str.upper(params['model2SIRD']) == 'DNN_FOURIERBASE':
                SNN_temp = DNN_base.DNN_FourierBase(T_it, Weight2S, Bias2S, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                INN_temp = DNN_base.DNN_FourierBase(T_it, Weight2I, Bias2I, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                RNN_temp = DNN_base.DNN_FourierBase(T_it, Weight2R, Bias2R, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                DNN_temp = DNN_base.DNN_FourierBase(T_it, Weight2D, Bias2D, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)

            freq2paras = params['freq2paras']
            if 'DNN' == str.upper(params['model2paras']):
                in_beta = DNN_base.DNN(T_it, Weight2beta, Bias2beta, hidden_para, activateIn_name=params['actIn_Name2paras'],
                                       activate_name=act_func2paras)
                in_gamma = DNN_base.DNN(T_it, Weight2gamma, Bias2gamma, hidden_para,
                                        activateIn_name=params['actIn_Name2paras'], activate_name=act_func2paras)
                in_mu = DNN_base.DNN(T_it, Weight2mu, Bias2mu, hidden_para,
                                     activateIn_name=params['actIn_Name2paras'], activate_name=act_func2paras)
            elif 'DNN_SCALE' == str.upper(params['model2paras']):
                in_beta = DNN_base.DNN_scale(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                             activateIn_name=params['actIn_Name2paras'], activate_name=act_func2paras)
                in_gamma = DNN_base.DNN_scale(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                              activateIn_name=params['actIn_Name2paras'], activate_name=act_func2paras)
                in_mu = DNN_base.DNN_scale(T_it, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                           activateIn_name=params['actIn_Name2paras'], activate_name=act_func2paras)
            elif str.upper(params['model2paras']) == 'DNN_FOURIERBASE':
                in_beta = DNN_base.DNN_FourierBase(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                   activate_name=act_func2paras, sFourier=1.0)
                in_gamma = DNN_base.DNN_FourierBase(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                    activate_name=act_func2paras, sFourier=1.0)
                in_mu = DNN_base.DNN_FourierBase(T_it, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                 activate_name=act_func2paras, sFourier=1.0)

            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            # 在归一化条件下: 如果总的“人口”和归一化"人口"的数值一致，这样的话，归一化后的数值会很小
            # tf.square(), tf.tanh(), tf.nn.relu(), tf.abs(), modelUtils.gauss()
            beta = tf.square(in_beta)
            # beta = tf.nn.sigmoid(in_beta)
            gamma = tf.nn.sigmoid(in_gamma)
            mu = tf.nn.sigmoid(in_mu)

            S_NN = tf.nn.sigmoid(SNN_temp)
            I_NN = tf.nn.sigmoid(INN_temp)
            R_NN = tf.nn.sigmoid(RNN_temp)
            D_NN = tf.nn.sigmoid(DNN_temp)

            N_NN = S_NN + I_NN + R_NN + D_NN

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dD_NN2t = tf.gradients(D_NN, T_it)[0]
            dN_NN2t = tf.gradients(N_NN, T_it)[0]

            temp_snn2t = -beta*S_NN*I_NN/(S_NN + I_NN)
            temp_inn2t = beta*S_NN*I_NN - gamma * I_NN - mu * I_NN
            temp_rnn2t = gamma *I_NN
            temp_dnn2t = mu * I_NN

            if str.lower(params['loss_function']) == 'l2_loss' and params['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(I_NN - I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(R_NN - R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(D_NN - D_observe))
                # LossN_Net_obs = tf.reduce_mean(tf.square(N_NN - N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(params['loss_function']) == 'l2_loss' and params['scale_up'] == 1:
                scale_up = params['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.square(scale_up*S_NN - scale_up*S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(scale_up*I_NN - scale_up*I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(scale_up*R_NN - scale_up*R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(scale_up*D_NN - scale_up*D_observe))
                # LossN_Net_obs = tf.reduce_mean(tf.square(scale_up*N_NN - scale_up*N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(params['loss_function']) == 'lncosh_loss' and params['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(S_NN - S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(I_NN - I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(R_NN - R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(D_NN - D_observe)))
                # LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(N_NN - N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
                Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))
            elif str.lower(params['loss_function']) == 'lncosh_loss' and params['scale_up'] == 1:
                scale_up = params['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(scale_up*S_NN - scale_up*S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*I_NN - scale_up*I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*R_NN - scale_up*R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*D_NN - scale_up*D_observe)))
                # LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*N_NN - scale_up*N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
                Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))

            if params['regular_weight_model'] == 'L1':
                regular_WB2S = DNN_base.regular_weights_biases_L1(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L1(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L1(Weight2D, Bias2D)
                regular_WB2Beta = DNN_base.regular_weights_biases_L1(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L1(Weight2gamma, Bias2gamma)
                regular_WB2Mu = DNN_base.regular_weights_biases_L1(Weight2mu, Bias2mu)
            elif params['regular_weight_model'] == 'L2':
                regular_WB2S = DNN_base.regular_weights_biases_L2(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L2(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L2(Weight2D, Bias2D)
                regular_WB2Beta = DNN_base.regular_weights_biases_L2(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L2(Weight2gamma, Bias2gamma)
                regular_WB2Mu = DNN_base.regular_weights_biases_L2(Weight2mu, Bias2mu)
            else:
                regular_WB2S = tf.constant(0.0)
                regular_WB2I = tf.constant(0.0)
                regular_WB2R = tf.constant(0.0)
                regular_WB2D = tf.constant(0.0)
                regular_WB2Beta = tf.constant(0.0)
                regular_WB2Gamma = tf.constant(0.0)
                regular_WB2Mu = tf.constant(0.0)

            PWB2S = wb_penalty*regular_WB2S
            PWB2I = wb_penalty*regular_WB2I
            PWB2R = wb_penalty*regular_WB2R
            PWB2D = wb_penalty * regular_WB2D
            PWB2Beta = wb_penalty * regular_WB2Beta
            PWB2Gamma = wb_penalty * regular_WB2Gamma
            PWB2Mu = wb_penalty * regular_WB2Mu

            # Loss2S = Loss2dS + PWB2S
            Loss2S = predict_true_penalty * LossS_Net_obs + Loss2dS + PWB2S
            # Loss2I = Loss2dI + PWB2I
            Loss2I = predict_true_penalty * LossI_Net_obs + Loss2dI + PWB2I
            # Loss2R = Loss2dR + PWB2R
            Loss2R = predict_true_penalty * LossR_Net_obs + Loss2dR + PWB2R
            # Loss2D = Loss2dD + PWB2D
            Loss2D = predict_true_penalty * LossD_Net_obs + Loss2dD + PWB2D

            Loss = Loss2S + Loss2I + Loss2R + Loss2D +  PWB2Beta + PWB2Gamma + PWB2Mu

            optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if params['train_model'] == 'train_group':
                train_Loss2S = optimizer.minimize(Loss2S, global_step=global_steps)
                train_Loss2I = optimizer.minimize(Loss2I, global_step=global_steps)
                train_Loss2R = optimizer.minimize(Loss2R, global_step=global_steps)
                train_Loss2D = optimizer.minimize(Loss2D, global_step=global_steps)
                # train_Loss2N = my_optimizer.minimize(Loss2N, global_step=global_steps)
                train_Losses = tf.group(train_Loss2S, train_Loss2I, train_Loss2R, train_Loss2D)
            elif params['train_model'] == 'train_union_loss':
                train_Losses = optimizer.minimize(Loss, global_step=global_steps)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_n_all, loss_all = [], [], [], [], [], []
    test_epoch = []
    test_mse2I_all, test_rel2I_all = [], []

    filename = FLAGS.data_fname
    data_list = dataUtils.load_data(filename, N=FLAGS.population)
    # 按顺序取出列表中的数据
    date, data2S, data2I, data2R, data2D, *_ = data_list
    train_data, test_data = dataUtils.split_data(date, data2S, data2I, data2R, data2D, train_size=1.0, normalFactor=FLAGS.population)
    train_date, train_data2s, train_data2i, train_data2r, train_data2d, *_ = train_data
    # test_date, test_data2s, test_data2i, test_data2r, test_data2d, *_ = test_data
    test_date, test_data2s, test_data2i, test_data2r, test_data2d, *_ = train_data  

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    batchSize_test = len(test_date)
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)

    # 由于将数据拆分为训练数据和测试数据时，进行了归一化处理，故这里不用归一化
    s_obs_test = DNN_data.sample_testData_serially(test_data2s, batchSize_test, normalFactor=1.0)
    i_obs_test = DNN_data.sample_testData_serially(test_data2i, batchSize_test, normalFactor=1.0)
    r_obs_test = DNN_data.sample_testData_serially(test_data2r, batchSize_test, normalFactor=1.0)
    d_obs_test = DNN_data.sample_testData_serially(test_data2d, batchSize_test, normalFactor=1.0)


    print('The test data about i:\n', str(np.transpose(i_obs_test)))
    print('\n')
    print('The test data about s:\n', str(np.transpose(s_obs_test)))
    print('\n')
    DNN_tools.log_string('The test data about i:\n%s\n' % str(np.transpose(i_obs_test)), log_fileout)
    DNN_tools.log_string('The test data about s:\n%s\n' % str(np.transpose(s_obs_test)), log_fileout)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True                        # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                            # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tmp_lr = init_lr
        for i_epoch in range(params['max_epoch'] + 1):
            t_batch, s_obs, i_obs, r_obs, d_obs = \
                dataUtils.sample_data(train_date, train_data2s, train_data2i, train_data2r, train_data2d,
                                        window_size=FLAGS.batach_size, sampling_opt=FLAGS.sample_method)
            # n_obs = nbatch2train.reshape(batchSize_train, 1)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if params['activate_stage_penalty'] == 1:
                if i_epoch < int(params['max_epoch'] / 10):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < int(params['max_epoch'] / 5):
                    temp_penalty_pt = 10 * pt_penalty_init
                elif i_epoch < int(params['max_epoch'] / 4):
                    temp_penalty_pt = 50 * pt_penalty_init
                elif i_epoch < int(params['max_epoch'] / 2):
                    temp_penalty_pt = 100 * pt_penalty_init
                elif i_epoch < int(3 * params['max_epoch'] / 4):
                    temp_penalty_pt = 200 * pt_penalty_init
                else:
                    temp_penalty_pt = 500 * pt_penalty_init
            elif params['activate_stage_penalty'] == 2:
                if i_epoch < int(params['max_epoch'] / 3):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < 2 * int(params['max_epoch'] / 3):
                    temp_penalty_pt = 10 * pt_penalty_init
                else:
                    temp_penalty_pt = 50 * pt_penalty_init
            else:
                temp_penalty_pt = pt_penalty_init

            _, loss_s, loss_i, loss_r, loss_d, loss, pwb2s, pwb2i, pwb2r, pwb2d = sess.run(
                [train_Losses, Loss2S, Loss2I, Loss2R, Loss2D, Loss, PWB2S, PWB2I, PWB2R, PWB2D],
                feed_dict={T_it: t_batch, S_observe: s_obs, I_observe: i_obs,R_observe: r_obs,D_observe: d_obs, in_learning_rate: tmp_lr,
                           predict_true_penalty: temp_penalty_pt})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            # loss_n_all.append(loss_n)
            loss_all.append(loss)

            if i_epoch % 1000 == 0:
                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的训练结果
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_pt, pwb2s, pwb2i, pwb2r, loss_s,
                                    loss_i, loss_r, loss_d, loss, log_out=log_fileout)

                s_nn2train, i_nn2train, r_nn2train, d_nn2train = sess.run(
                    [S_NN, I_NN, R_NN, D_NN], feed_dict={T_it: np.reshape(train_date, [-1, 1])})

                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的测试结果
                test_epoch.append(i_epoch / 1000)
                s_nn2test, i_nn2test, r_nn2test, d_nn2test, beta_test, gamma_test, mu_test = sess.run(
                    [S_NN, I_NN, R_NN, D_NN, beta, gamma, mu], feed_dict={T_it: test_t_bach})
                point_ERR2I = np.square(i_nn2test - i_obs_test)
                test_mse2I = np.mean(point_ERR2I)
                test_mse2I_all.append(test_mse2I)
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2I_all.append(test_rel2I)

                DNN_tools.print_and_log_test_one_epoch(test_mse2I, test_rel2I, log_out=log_fileout)
                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch),
                                     log2testSolus)
                DNN_tools.log_string('The test result for s:\n%s\n' % str(np.transpose(s_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for i:\n%s\n' % str(np.transpose(i_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for r:\n%s\n\n' % str(np.transpose(r_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for d:\n%s\n\n' % str(np.transpose(d_nn2test)), log2testSolus)

                # --------以下代码为输出训练过程中 S_NN_temp, I_NN_temp, R_NN_temp, in_beta, in_gamma 的测试结果-------------
                s_nn_temp2test, i_nn_temp2test, r_nn_temp2test, d_nn_temp2test, in_beta_test, in_gamma_test = sess.run(
                    [SNN_temp, INN_temp, RNN_temp, DNN_temp, in_beta, in_gamma],
                    feed_dict={T_it: test_t_bach})

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for s_temp:\n%s\n' % str(np.transpose(s_nn_temp2test)),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for i_temp:\n%s\n' % str(np.transpose(i_nn_temp2test)),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for r_temp:\n%s\n\n' % str(np.transpose(r_nn_temp2test)),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for d_temp:\n%s\n\n' % str(np.transpose(d_nn_temp2test)),
                                     log2testSolus2)

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch),
                                     log2testParas)
                DNN_tools.log_string('The test result for in_beta:\n%s\n' % str(np.transpose(in_beta_test)),
                                     log2testParas)
                DNN_tools.log_string('The test result for in_gamma:\n%s\n' % str(np.transpose(in_gamma_test)),
                                     log2testParas)

        # 把groud truth 和 predict数据存储在csv文件里
        data_dic = {'test_data2s': test_data2s, 's_nn2test': np.squeeze(s_nn2test),
                    'test_data2i': test_data2i, 'i_nn2test': np.squeeze(i_nn2test),
                    'test_data2r': test_data2r, 'r_nn2test': np.squeeze(r_nn2test),
                    'test_data2d': test_data2d, 'd_nn2test': np.squeeze(d_nn2test)
        }
        import pandas as pd
        data_df = pd.DataFrame.from_dict(data_dic)
        data_df.to_csv(params['FolderName'] + '/sird_results.csv', index = False)
        # parameters
        paras_dic = {'beta2test': np.squeeze(beta_test), 
                'gamma2test': np.squeeze(gamma_test), 
                'mu2test': np.squeeze(mu_test)}
        paras_df = pd.DataFrame.from_dict(paras_dic)
        paras_df.to_csv(params['FolderName'] + '/params_results.csv', index = False)
        # save loss data
        loss_dic = {'loss_s':loss_s_all,
                    'loss_i':loss_i_all,
                    'loss_r':loss_r_all,
                    'loss_d':loss_d_all}
        loss_df = pd.DataFrame.from_dict(loss_dic)
        loss_df.to_csv(params['FolderName'] + '/loss_results.csv', index = False)

        DNN_tools.log_string('The train result for S:\n%s\n' % str(np.transpose(s_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for I:\n%s\n' % str(np.transpose(i_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for params:\n%s\n\n' % str(np.transpose(r_nn2train)), log2trianSolus)

        # saveData.true_value2convid(train_data2i, name2Array='itrue2train', outPath=params['FolderName'])
        # saveData.save_Solu2mat_Covid(s_nn2train, name2solus='s2train', outPath=params['FolderName'])
        # saveData.save_Solu2mat_Covid(i_nn2train, name2solus='i2train', outPath=params['FolderName'])
        # saveData.save_Solu2mat_Covid(r_nn2train, name2solus='r2train', outPath=params['FolderName'])
        # saveData.save_Solu2mat_Covid(d_nn2train, name2solus='d2train', outPath=params['FolderName'])

        # saveData.save_SIR_trainLoss2mat_Covid(loss_s_all, loss_i_all, loss_r_all, loss_n_all, actName=act_func2SIRD,
        #                                       outPath=params['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=params['seed'], outPath=params['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=params['seed'], outPath=params['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=params['seed'], outPath=params['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_n_all, lossType='loss2n', seedNo=params['seed'], outPath=params['FolderName'],
                                          yaxis_scale=True)

        # saveData.true_value2convid(i_obs_test, name2Array='i_true2test', outPath=params['FolderName'])
        # saveData.save_testMSE_REL2mat(test_mse2I_all, test_rel2I_all, actName='Infected', outPath=params['FolderName'])
        plotData.plotTest_MSE_REL(test_mse2I_all, test_rel2I_all, test_epoch, actName='Infected', seedNo=params['seed'],
                                  outPath=params['FolderName'], yaxis_scale=True)
        # saveData.save_SIR_testSolus2mat_Covid(s_nn2test, i_nn2test, r_nn2test, name2solus1='snn2test',
        #                                       name2solus2='inn2test', name2solus3='rnn2test', outPath=params['FolderName'])
        # saveData.save_SIR_testParas2mat_Covid(beta_test, gamma_test, name2para1='beta2test', name2para2='gamma2test',
        #                                       outPath=params['FolderName'])

        # plotData.plot_testSolu2convid(s_nn2test, name2solu='s_test', coord_points2test=test_t_bach,
        #                               outPath=params['FolderName'])
        # plotData.plot_testSolu2convid(i_obs_test, name2solu='i_true', coord_points2test=test_t_bach,
        #                               outPath=params['FolderName'])
        # plotData.plot_testSolu2convid(i_nn2test, name2solu='i_test', coord_points2test=test_t_bach,
        #                               outPath=params['FolderName'])
        # plotData.plot_testSolu2convid(r_nn2test, name2solu='r_test', coord_points2test=test_t_bach,
        #                               outPath=params['FolderName'])

        plotData.plot_testSolus2convid(s_obs_test, s_nn2test, name2solu1='s_true', name2solu2='s_test',
                                       coord_points2test=test_t_bach, seedNo=params['seed'], outPath=params['FolderName'])
        plotData.plot_testSolus2convid(i_obs_test, i_nn2test, name2solu1='i_true', name2solu2='i_test',
                                       coord_points2test=test_t_bach, seedNo=params['seed'], outPath=params['FolderName'])
        plotData.plot_testSolus2convid(r_obs_test, r_nn2test, name2solu1='r_true', name2solu2='r_test',
                                       coord_points2test=test_t_bach, seedNo=params['seed'], outPath=params['FolderName'])
        plotData.plot_testSolus2convid(d_obs_test, d_nn2test, name2solu1='d_true', name2solu2='d_test',
                                       coord_points2test=test_t_bach, seedNo=params['seed'], outPath=params['FolderName'])

        plotData.plot_testSolu2convid(beta_test, name2solu='beta_test', coord_points2test=test_t_bach,
                                      outPath=params['FolderName'])
        plotData.plot_testSolu2convid(gamma_test, name2solu='gamma_test', coord_points2test=test_t_bach,
                                      outPath=params['FolderName'])
        plotData.plot_testSolu2convid(mu_test, name2solu='mu_test', coord_points2test=test_t_bach,
                                      outPath=params['FolderName'])

def main(unused_argv):
    params = {}
    params['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'output_SIRD'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    params['seed'] = np.random.randint(1e5)
    seed_str = str(params['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    params['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    params['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    params['max_epoch'] = 200000
    if 0 != params['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        params['max_epoch'] = int(epoch_stop)

    # ----------------------------------------- Convid 设置 ---------------------------------
    params['eqs_name'] = 'SIRD'
    params['input_dim'] = 1                       # 输入维数，即问题的维数(几元问题)
    params['output_dim'] = 1                      # 输出维数
    params['total_population'] = 3450000          # 总的“人口”数量

    # params['normalize_population'] = 3450000     # 归一化时使用的“人口”数值
    # params['normalize_population'] = 100000
    # params['normalize_population'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    params['size2train'] = 70                    # 训练集的大小
    params['batch_size2train'] = 20              # 训练数据的批大小
    params['batch_size2test'] = 10               # 训练数据的批大小
    # params['opt2sample'] = 'random_sample'     # 训练集的选取方式--随机采样
    # params['opt2sample'] = 'rand_sample_sort'    # 训练集的选取方式--随机采样后按时间排序
    params['opt2sample'] = 'windows_rand_sample'  # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    params['init_penalty2predict_true'] = 50     # Regularization parameter for boundary conditions
    params['activate_stage_penalty'] = 1         # 是否开启阶段调整惩罚项，0 代表不调整，非 0 代表调整
    if params['activate_stage_penalty'] == 1 or params['activate_stage_penalty'] == 2:
        # params['init_penalty2predict_true'] = 1000
        # params['init_penalty2predict_true'] = 100
        # params['init_penalty2predict_true'] = 50
        # params['init_penalty2predict_true'] = 20
        params['init_penalty2predict_true'] = 1

    # params['regular_weight_model'] = 'L0'
    # params['regular_weight'] = 0.000             # Regularization parameter for weights

    # params['regular_weight_model'] = 'L1'
    params['regular_weight_model'] = 'L2'          # The model of regular weights and biases
    # params['regular_weight'] = 0.001             # Regularization parameter for weights
    # params['regular_weight'] = 0.0005            # Regularization parameter for weights
    # params['regular_weight'] = 0.0001            # Regularization parameter for weights
    params['regular_weight'] = 0.00005             # Regularization parameter for weights
    # params['regular_weight'] = 0.00001           # Regularization parameter for weights

    params['optimizer_name'] = 'Adam'              # 优化器
    params['loss_function'] = 'L2_loss'            # 损失函数的类型
    # params['loss_function'] = 'lncosh_loss'      # 损失函数的类型
    params['scale_up'] = 1                         # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    params['scale_factor'] = 100                   # scale_factor 用来对数值进行尺度提升，如1e-6量级提升到1e-2量级

    # params['train_model'] = 'train_group'        # 训练模式:各个不同的loss捆绑打包训练
    params['train_model'] = 'train_union_loss'     # 训练模式:各个不同的loss累加在一起，训练

    if 50000 < params['max_epoch']:
        params['learning_rate'] = 2e-3             # 学习率
        params['lr_decay'] = 1e-4                  # 学习率 decay
        # params['learning_rate'] = 2e-4           # 学习率
        # params['lr_decay'] = 5e-5                # 学习率 decay
    elif (20000 < params['max_epoch'] and 50000 >= params['max_epoch']):
        # params['learning_rate'] = 1e-3           # 学习率
        # params['lr_decay'] = 1e-4                # 学习率 decay
        # params['learning_rate'] = 2e-4           # 学习率
        # params['lr_decay'] = 1e-4                # 学习率 decay
        params['learning_rate'] = 1e-4             # 学习率
        params['lr_decay'] = 5e-5                  # 学习率 decay
    else:
        params['learning_rate'] = 5e-5             # 学习率
        params['lr_decay'] = 1e-5                  # 学习率 decay

    # SIRD和参数网络模型的选择
    # params['model2SIRD'] = 'DNN'
    # params['model2SIRD'] = 'DNN_scale'
    # params['model2SIRD'] = 'DNN_scaleOut'
    params['model2SIRD'] = 'DNN_FourierBase'

    # params['model2paras'] = 'DNN'
    # params['model2paras'] = 'DNN_scale'
    # params['model2paras'] = 'DNN_scaleOut'
    params['model2paras'] = 'DNN_FourierBase'

    # SIRD和参数网络模型的隐藏层单元数目
    if params['model2SIRD'] == 'DNN_FourierBase':
        params['hidden2SIRD'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # params['hidden2SIRD'] = (10, 10, 8, 6, 6, 3)        # it is used to debug our work
        params['hidden2SIRD'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # params['hidden2SIRD'] = (80, 80, 60, 40, 40, 20)    # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # params['hidden2SIRD'] = (100, 100, 80, 60, 60, 40)
        # params['hidden2SIRD'] = (200, 100, 100, 80, 50, 50)

    if params['model2paras'] == 'DNN_FourierBase':
        params['hidden2para'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # params['hidden2para'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        params['hidden2para'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # params['hidden2para'] = (80, 80, 60, 40, 40, 20)   # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # params['hidden2para'] = (100, 100, 80, 60, 60, 40)
        # params['hidden2para'] = (200, 100, 100, 80, 50, 50)

    # SIRD和参数网络模型的尺度因子
    if params['model2SIRD'] != 'DNN':
        params['freq2SIRD'] = np.concatenate(([1], np.arange(1, 20)), axis=0)
    if params['model2paras'] != 'DNN':
        params['freq2paras'] = np.concatenate(([1], np.arange(1, 20)), axis=0)

    # SIRD和参数网络模型为傅里叶网络和尺度网络时，重复高频因子或者低频因子
    if params['model2SIRD'] == 'DNN_FourierBase' or params['model2SIRD'] == 'DNN_scale':
        params['if_repeat_High_freq2SIRD'] = False
    if params['model2paras'] == 'DNN_FourierBase' or params['model2paras'] == 'DNN_scale':
        params['if_repeat_High_freq2paras'] = False

    # SIRD和参数网络模型的激活函数的选择
    params['actIn_Name2SIRD'] = 'tanh'

    params['act_Name2SIRD'] = 'tanh'  # 这个激活函数比较s2ReLU合适

    params['actIn_Name2paras'] = 'tanh'

    params['act_Name2paras'] = 'tanh'  # 这个激活函数比较s2ReLU合适

    solve_SIRD2COVID(params)

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)