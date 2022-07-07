from enum import Flag
import os
from sqlite3 import paramstyle
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import checkmate 

from datetime import datetime
import argparse
import pandas as pd

import dataUtils
import modelUtils
import matplotlib.pyplot as plt

# tf2兼容tf1
tf.compat.v1.disable_eager_execution()

# 在模型中除以N

DECAY_STEPS = 20000
DECAY_RATE = 0.9

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

parser.add_argument('--best_model_dir', type=str, default='./models',
                    help='.' '0.1, 0.01, 0.05, 0,001')


def solve_SIRD2COVID(params):

    if not os.path.exists(params['FolderName']):  # 判断路径是否存在
        os.mkdir(params['FolderName']) 
    
    # 记录训练日志
    log_fileout = open(os.path.join(params['FolderName'], 'log_train.txt'), 'w')
    # dataUtils.dictionary_out2file2(params, log_fileout)

    # 网络初始化
    if params['sird_network'].upper() == 'DNN_FOURIERBASE':
        init_func = modelUtils.Xavier_init_NN_Fourier
    else:
        init_func = modelUtils.Xavier_init_NN
    Weight2S, Bias2S = init_func(params['input_dim'], params['output_dim'], params['hidden_sird'], 'wb_s')
    Weight2I, Bias2I = init_func(params['input_dim'], params['output_dim'], params['hidden_sird'], 'wb_i')
    Weight2R, Bias2R = init_func(params['input_dim'], params['output_dim'], params['hidden_sird'], 'wb_r')
    Weight2D, Bias2D = init_func(params['input_dim'], params['output_dim'], params['hidden_sird'], 'wb_d')

    if params['params_network'].upper() == 'DNN_FOURIERBASE':
        init_func = modelUtils.Xavier_init_NN_Fourier
    else:
        init_func = modelUtils.Xavier_init_NN
    Weight2beta, Bias2beta = init_func(params['input_dim'], params['output_dim'], params['hidden_params'], 'wb_beta')
    Weight2gamma, Bias2gamma = init_func(params['input_dim'], params['output_dim'], params['hidden_params'], 'wb_gamma')
    Weight2mu, Bias2mu = init_func(params['input_dim'], params['output_dim'], params['hidden_params'], 'wb_mu')

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (params['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            # 占位符
            T_it = tf.compat.v1.placeholder(tf.float32, name='T_it', shape=[None, params['output_dim']])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[None, params['output_dim']])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[None, params['output_dim']])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[None, params['output_dim']])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[None, params['output_dim']])
            predict_true_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='pt_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            freq2SIRD = params['freq2SIRD']
            # 网络结构 
            with tf.name_scope('sird_network'):
                if params['sird_network'].upper() == 'DNN':
                    SNN_temp = modelUtils.DNN(T_it, Weight2S, Bias2S, params['hidden_sird'], activateIn_name=params['activateIn_sird'],
                                            activate_name=params['activate_sird'])
                    INN_temp = modelUtils.DNN(T_it, Weight2I, Bias2I, params['hidden_sird'], activateIn_name=params['activateIn_sird'],
                                            activate_name=params['activate_sird'])
                    RNN_temp = modelUtils.DNN(T_it, Weight2R, Bias2R, params['hidden_sird'], activateIn_name=params['activateIn_sird'],
                                            activate_name=params['activate_sird'])
                    DNN_temp = modelUtils.DNN(T_it, Weight2D, Bias2D, params['hidden_sird'], activateIn_name=params['activateIn_sird'],
                                            activate_name=params['activate_sird'])

                elif params['sird_network'].upper() == 'DNN_SCALE':
                    SNN_temp = modelUtils.DNN_scale(T_it, Weight2S, Bias2S, params['hidden_sird'], freq2SIRD,
                                                activateIn_name=params['activateIn_sird'], activate_name=params['activate_sird'])
                    INN_temp = modelUtils.DNN_scale(T_it, Weight2I, Bias2I, params['hidden_sird'], freq2SIRD,
                                                activateIn_name=params['activateIn_sird'], activate_name=params['activate_sird'])
                    RNN_temp = modelUtils.DNN_scale(T_it, Weight2R, Bias2R, params['hidden_sird'], freq2SIRD,
                                                activateIn_name=params['activateIn_sird'], activate_name=params['activate_sird'])
                    DNN_temp = modelUtils.DNN_scale(T_it, Weight2D, Bias2D, params['hidden_sird'], freq2SIRD,
                                                activateIn_name=params['activateIn_sird'], activate_name=params['activate_sird'])
                elif params['sird_network'].upper() == 'DNN_FOURIERBASE':
                    SNN_temp = modelUtils.DNN_FourierBase(T_it, Weight2S, Bias2S, params['hidden_sird'], freq2SIRD,
                                                        activate_name=params['activate_sird'], sFourier=1.0)
                    INN_temp = modelUtils.DNN_FourierBase(T_it, Weight2I, Bias2I, params['hidden_sird'], freq2SIRD,
                                                        activate_name=params['activate_sird'], sFourier=1.0)
                    RNN_temp = modelUtils.DNN_FourierBase(T_it, Weight2R, Bias2R, params['hidden_sird'], freq2SIRD,
                                                        activate_name=params['activate_sird'], sFourier=1.0)
                    DNN_temp = modelUtils.DNN_FourierBase(T_it, Weight2D, Bias2D, params['hidden_sird'], freq2SIRD,
                                                        activate_name=params['activate_sird'], sFourier=1.0)

            freq2paras = params['freq2paras']
            with tf.name_scope('paras_network'):
                if 'DNN' == str.upper(params['params_network']):
                    in_beta = modelUtils.DNN(T_it, Weight2beta, Bias2beta, params['hidden_params'],
                                            activateIn_name=params['activateIn_params'], ctivate_name=params['activate_params'])
                    in_gamma = modelUtils.DNN(T_it, Weight2gamma, Bias2gamma, params['hidden_params'],
                                            activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])
                    in_mu = modelUtils.DNN(T_it, Weight2mu, Bias2mu, params['hidden_params'],
                                            activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])

                elif 'DNN_SCALE' == str.upper(params['params_network']):
                    in_beta = modelUtils.DNN_scale(T_it, Weight2beta, Bias2beta, params['hidden_params'], freq2paras,
                                                activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])
                    in_gamma = modelUtils.DNN_scale(T_it, Weight2gamma, Bias2gamma, params['hidden_params'], freq2paras,
                                                activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])
                    in_mu = modelUtils.DNN_scale(T_it, Weight2mu, Bias2mu, params['hidden_params'], freq2paras,
                                                activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])                                              
                elif str.upper(params['sird_network']) == 'DNN_FOURIERBASE':
                    in_beta = modelUtils.DNN_FourierBase(T_it, Weight2beta, Bias2beta, params['hidden_params'], freq2paras,
                                                    activate_name=params['activate_params'], sFourier=1.0)
                    in_gamma = modelUtils.DNN_FourierBase(T_it, Weight2gamma, Bias2gamma, params['hidden_params'], freq2paras,
                                                        activate_name=params['activate_params'], sFourier=1.0)
                    in_mu = modelUtils.DNN_FourierBase(T_it, Weight2mu, Bias2mu, params['hidden_params'], freq2paras,
                                                        activate_name=params['activate_params'], sFourier=1.0)

            # 激活函数
            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            # 在归一化条件下: 如果总的“人口”和归一化"人口"的数值一致，这样的话，归一化后的数值会很小
            # tf.square(), tf.tanh(), tf.nn.relu(), tf.abs(), modelUtils.gauss()
            beta = tf.square(in_beta)
            # beta = tf.nn.sigmoid(in_beta)
            gamma = tf.nn.sigmoid(in_gamma)
            mu = 0.1 * tf.nn.sigmoid(in_mu)
            
            S_NN = tf.nn.sigmoid(SNN_temp)
            I_NN = tf.nn.sigmoid(INN_temp)
            R_NN = tf.nn.sigmoid(RNN_temp)
            D_NN = tf.nn.sigmoid(DNN_temp)

            # 求导，取出导数形式
            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dD_NN2t = tf.gradients(D_NN, T_it)[0]

            # 微分方程
            with tf.name_scope('ODEs'):
                temp_snn2t = - (beta*S_NN*I_NN) / (S_NN + I_NN)
                temp_inn2t = - (beta*S_NN*I_NN) / (S_NN + I_NN) - gamma * I_NN - mu * I_NN
                temp_rnn2t = gamma *I_NN
                temp_dnn2t = mu * I_NN

            # Loss function
            if params['loss_function'].lower() == 'l2_loss' and params['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(I_NN - I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(R_NN - R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(D_NN - D_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif params['loss_function'].lower() == 'l2_loss' and params['scale_up'] == 1:
                scale_up = params['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.square(scale_up*S_NN - scale_up*S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(scale_up*I_NN - scale_up*I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(scale_up*R_NN - scale_up*R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(scale_up*D_NN - scale_up*D_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif params['loss_function'].lower() == 'lncosh_loss' and params['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(S_NN - S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(I_NN - I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(R_NN - R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(D_NN - D_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
            elif params['loss_function'].lower() == 'lncosh_loss' and params['scale_up'] == 1:
                scale_up = params['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(scale_up*S_NN - scale_up*S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*I_NN - scale_up*I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*R_NN - scale_up*R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*D_NN - scale_up*D_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))

            # 正则化
            regular_func = lambda a, b: tf.constant(0.0)
            if params['regular_method'] == 'L1':
                regular_func = modelUtils.regular_weights_biases_L1
            elif params['regular_method'] == 'L2':
                regular_func = modelUtils.regular_weights_biases_L2
            regular_WB2S = regular_func(Weight2S, Bias2S)
            regular_WB2I = regular_func(Weight2I, Bias2I)
            regular_WB2R = regular_func(Weight2R, Bias2R)
            regular_WB2D = regular_func(Weight2D, Bias2D)
            regular_WB2Beta = regular_func(Weight2beta, Bias2beta)
            regular_WB2Gamma = regular_func(Weight2gamma, Bias2gamma)
            regular_WB2Mu = regular_func(Weight2mu, Bias2mu) 

            PWB2S = FLAGS.regular_weight * regular_WB2S
            PWB2I = FLAGS.regular_weight * regular_WB2I
            PWB2R = FLAGS.regular_weight * regular_WB2R
            PWB2D = FLAGS.regular_weight * regular_WB2D
            PWB2Beta = FLAGS.regular_weight * regular_WB2Beta
            PWB2Gamma = FLAGS.regular_weight * regular_WB2Gamma
            PWB2Mu = FLAGS.regular_weight * regular_WB2Mu

            # 定义loss
            with tf.name_scope('loss'):
                Loss2S = Loss2dS + PWB2S
                # Loss2S = predict_true_penalty * LossS_Net_obs + Loss2dS + PWB2S
                # Loss2I = Loss2dI + PWB2I
                Loss2I = predict_true_penalty * LossI_Net_obs + Loss2dI + PWB2I
                Loss2R = Loss2dR + PWB2R
                # Loss2R = predict_true_penalty * LossR_Net_obs + Loss2dR + PWB2R
                Loss2D = Loss2dD + PWB2D
                # Loss2D = predict_true_penalty * LossD_Net_obs + Loss2dD + PWB2D

                Loss = Loss2S + Loss2I + Loss2R + Loss2D + PWB2Beta + PWB2Gamma + PWB2Mu

            # 定义optimizer
            with tf.name_scope('optimizer'):
                optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
                if params['train_model'] == 'train_group':
                    train_Loss2S = optimizer.minimize(Loss2S, global_step=global_steps)
                    train_Loss2I = optimizer.minimize(Loss2I, global_step=global_steps)
                    train_Loss2R = optimizer.minimize(Loss2R, global_step=global_steps)
                    train_Loss2D = optimizer.minimize(Loss2D, global_step=global_steps)
                    train_Losses = tf.group(train_Loss2S, train_Loss2I, train_Loss2R, train_Loss2D)
                elif params['train_model'] == 'train_union_loss':
                    train_Losses = optimizer.minimize(Loss, global_step=global_steps)

    loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_all = [], [], [], [], []

    # test_epoch = []
    test_mse2S_all, test_rel2S_all = list(), list()
    test_mse2I_all, test_rel2I_all = list(), list()
    test_mse2R_all, test_rel2R_all = list(), list()
    test_mse2D_all, test_rel2D_all = list(), list()

    data_list = dataUtils.load_data(FLAGS.data_fname, N=3450000)
    date, data2S, data2I, data2R, data2D, *_ = data_list

    # 是否归一化数据，取决于normalFactor的值
    train_data, test_data = dataUtils.split_data(date, data2S, data2I, data2R, data2D, train_size=1.0, normalFactor=params['normalize_population'])
    # 按顺序取出列表中的数据
    train_date, train_data2s, train_data2i, train_data2r, train_data2d, *_ = train_data
    # test_date, test_data2s, test_data2i, test_data2r, test_data2d, *_ = test_data
    test_date, test_data2s, test_data2i, test_data2r, test_data2d, *_ = train_data  

    # 指定GPU0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 打印可用的GPU
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True                        # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                            # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    # 对学习率进行指数衰减
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate = FLAGS.initial_learning_rate,
                                                        global_step = global_steps,
                                                        decay_steps = DECAY_STEPS,
                                                        decay_rate = DECAY_RATE,
                                                        staircase = False)
    lr_list = list()
    # 某一范围内指定一个值
    # Multiply the penalty decay rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(FLAGS.train_epoches / 10), int(FLAGS.train_epoches / 5), int(FLAGS.train_epoches / 4),\
                int(FLAGS.train_epoches / 2), int(3 * FLAGS.train_epoches / 4)]
    values_arr = [1, 10, 50, 100, 200, 500]
    values = [params['init_penalty2predict_true'] * decay for decay in values_arr]    
    temp_predict_true_penalty = tf.compat.v1.train.piecewise_constant(
                                tf.cast(global_steps, tf.int32), boundaries, values)
    # Training   
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # best_ckpt_saver = checkmate.BestCheckpointSaver(save_dir=os.path.join(params['FolderName'], './models'),
                                            # num_to_keep=3,
                                            # maximize=False)
        for epoch in range(FLAGS.train_epoches):
            t_batch, s_obs, i_obs, r_obs, d_obs = \
                dataUtils.sample_data(train_date, train_data2s, train_data2i, train_data2r, train_data2d,
                                        window_size=FLAGS.batach_size, sampling_opt=params['sample_method'])
            in_learning_rate = sess.run(learning_rate, feed_dict={global_steps:epoch})
            lr_list.append(in_learning_rate)
            predict_true_penalty = sess.run(temp_predict_true_penalty, feed_dict={global_steps:epoch}) 

            _, loss_s, loss_i, loss_r, loss_d, loss, pwb2s, pwb2i, pwb2r, pwb2d = sess.run(
                [train_Losses, Loss2S, Loss2I, Loss2R, Loss2D, Loss, PWB2S, PWB2I, PWB2R, PWB2D],
                feed_dict={T_it: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            loss_all.append(loss)

            # 每10000轮测试一次
            mse_lowest = 1.0
            if epoch % FLAGS.epoches_per_eval == 0:
                print('-----------------', epoch, '-----------------------')
                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, D_NN, beta, gamma 的训练结果
                s_nn2test, i_nn2test, r_nn2test, d_nn2test = sess.run(
                    [S_NN, I_NN, R_NN, D_NN], feed_dict={T_it: np.reshape(test_date, [-1, 1])})
                beta2test, gamma2test, mu2test = sess.run(
                    [beta, gamma, mu], feed_dict={T_it: np.reshape(test_date, [-1, 1])})
                print(beta2test,'--', gamma2test,'--', mu2test)
                # best_ckpt_saver.handle(mse_average, sess, global_steps) 

                # test_epoch.append(epoch / 1000)

                test_mse2S, test_rel2S = dataUtils.compute_mse_res(s_nn2test, test_data2s)
                test_mse2I, test_rel2I = dataUtils.compute_mse_res(i_nn2test, test_data2i)
                test_mse2R, test_rel2R = dataUtils.compute_mse_res(r_nn2test, test_data2r)
                test_mse2D, test_rel2D = dataUtils.compute_mse_res(d_nn2test, test_data2d)
                test_mse2S_all.append(test_mse2S)
                test_rel2S_all.append(test_rel2S)
                test_mse2I_all.append(test_mse2I)
                test_rel2I_all.append(test_rel2I)
                test_mse2R_all.append(test_mse2R)
                test_rel2R_all.append(test_rel2R)
                test_mse2D_all.append(test_mse2D)
                test_rel2D_all.append(test_rel2D)

                mse_list = [test_mse2S, test_mse2I, test_mse2R, test_mse2D]
                mse_average = sum(mse_list) / len(mse_list)

                if mse_average < mse_lowest:
                    mse_lowest = mse_average
                    if epoch > 100000:
                        # 把groud truth 和 predict数据存储在csv文件里
                        data_dic = {'test_data2s': test_data2s, 's_nn2test': np.squeeze(s_nn2test),
                                    'test_data2i': test_data2i, 'i_nn2test': np.squeeze(i_nn2test),
                                    'test_data2r': test_data2r, 'r_nn2test': np.squeeze(r_nn2test),
                                    'test_data2d': test_data2d, 'd_nn2test': np.squeeze(d_nn2test)
                        }
                        data_df = pd.DataFrame.from_dict(data_dic)
                        data_df.to_csv(params['FolderName'] + '/sird_results.csv', index = False)
                        # parameters
                        paras_dic = {'beta2test': np.squeeze(beta2test), 
                                'gamma2test': np.squeeze(gamma2test), 
                                'mu2test': np.squeeze(mu2test)}
                        paras_df = pd.DataFrame.from_dict(paras_dic)
                        paras_df.to_csv(params['FolderName'] + '/params_results.csv', index = False)
                        # save loss data
                        loss_dic = {'loss_s':loss_s_all,
                                    'loss_i':loss_i_all,
                                    'loss_r':loss_r_all,
                                    'loss_d':loss_d_all,
                                    'lr': lr_list}
                        loss_df = pd.DataFrame.from_dict(loss_dic)
                        loss_df.to_csv(params['FolderName'] + '/loss_results.csv', index = False)

                        break 
        
    # plt.figure(1)
    # plt.plot(range(FLAGS.train_epoches), lr_list, 'r-')
    # plt.savefig('learning_rate.png')

def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    params={ #'data_fname': FLAGS.data_fname,
            'output_dir': FLAGS.output_dir,
            'optimizer': FLAGS.optimizer,
            #'train_epoches': FLAGS.train_epoches,
            'eqs_name': FLAGS.eqs_name,
            'batach_size': FLAGS.batach_size,
            'sample_method': FLAGS.sample_method,
            'sird_network': FLAGS.sird_network,
            'params_network': FLAGS.params_network,
            'hidden_sird': FLAGS.hidden_sird,
            'hidden_params': FLAGS.hidden_params,
            'loss_function': FLAGS.loss_function,
            'activateIn_sird': FLAGS.activateIn_sird,
            'activate_sird': FLAGS.activate_sird,
            'activateIn_params': FLAGS.activateIn_params,
            'activate_params': FLAGS.activate_params,
            'init_penalty2predict_true': FLAGS.init_penalty2predict_true,
            'activate_stage_penalty': FLAGS.activate_stage_penalty,
            'regular_method': FLAGS.regular_method,
            'regular_weight': FLAGS.regular_weight,
            'input_dim': 1, # 输入维数，即问题的维数(几元问题)
            'output_dim': 1,  # 输出维数
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'total_population': 3450000,
    }

    params['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, FLAGS.output_dir)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    params['seed'] = 43
    timeFolder = datetime.now().strftime("%Y%m%d_%H%M")       # 当前时间为文件夹名
    params['FolderName'] = os.path.join(OUT_DIR, timeFolder)  # 路径连接
    FolderName = params['FolderName']
    if not os.path.exists(FolderName):
        os.makedirs(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ----------------------------------------- Convid 设置 ---------------------------------
    params['normalize_population'] = 3450000 #9776000      # 归一化时使用的“人口”数值
    # params['batch_size2test'] = 10 
    # ------------------------------------  神经网络的设置  ----------------------------------------
    # 是否开启阶段调整边界惩罚项 , 0 代表不调整，非 0 代表调整
    if FLAGS.activate_stage_penalty == True:
        params['init_penalty2predict_true'] = 50   # Regularization parameter for boundary conditions
    else:
        params['init_penalty2predict_true'] = 1


    params['scale_up'] = 1                         # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    params['scale_factor'] = 100                   # scale_factor 用来对数值进行尺度提升，如1e-6量级提升到1e-2量级

    # params['train_model'] = 'train_group'        # 训练模式:各个不同的loss捆绑打包训练
    params['train_model'] = 'train_union_loss'     # 训练模式:各个不同的loss累加在一起，训练

    # SIRD和参数网络模型的隐藏层单元数目
    if params['sird_network'] == 'DNN_FourierBase':
        params['hidden_sird'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # [(70, 50, 30, 30, 20), (80, 80, 60, 40, 40, 20), (100, 100, 80, 60, 60, 40), (200, 100, 100, 80, 50, 50)]
        params['hidden_sird'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570

    if params['params_network'] == 'DNN_FourierBase':
        params['hidden_params'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # [(70, 50, 30, 30, 20), (80, 80, 60, 40, 40, 20), (100, 100, 80, 60, 60, 40), (200, 100, 100, 80, 50, 50)]
        params['hidden_params'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570

    # SIRD和参数网络模型的尺度因子
    if params['sird_network'] != 'DNN':
        params['freq2SIRD'] = np.concatenate(([1], np.arange(1, 20)), axis=0)
    if params['sird_network'] != 'DNN':
        params['freq2paras'] = np.concatenate(([1], np.arange(1, 20)), axis=0)

    # # SIRD和参数网络模型为傅里叶网络和尺度网络时，重复高频因子或者低频因子
    # # 好像并没有用到这几行代码？？？
    # if params['sird_network'] == 'DNN_FourierBase' or params['sird_network'] == 'DNN_scale':
    #     params['if_repeat_High_freq2SIRD'] = False
    # if params['params_network'] == 'DNN_FourierBase' or params['params_network'] == 'DNN_scale':
    #     params['if_repeat_High_freq2paras'] = False

    solve_SIRD2COVID(params)




if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)


