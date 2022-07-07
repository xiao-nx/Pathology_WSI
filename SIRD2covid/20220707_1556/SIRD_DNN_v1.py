"""
@author: Xiao Ning
Benchmark Code of SIRD model
2022-06-30
"""
import os
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import modelUtils
import DNN_data

import pandas as pd
from datetime import datetime

import dataUtils
import modelUtils
# tf2兼容tf1
tf.compat.v1.disable_eager_execution()

def solve_SIRD2COVID(Params):
    log_out_path = Params['FolderName']        # 将路径从字典 Params 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    # dataUtils.dictionary_out2file(Params, log_fileout)

    log2trianSolus = open(os.path.join(log_out_path, 'train_Solus.txt'), 'w')      # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus = open(os.path.join(log_out_path, 'test_Solus.txt'), 'w')        # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus2 = open(os.path.join(log_out_path, 'test_Solus_temp.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testParas = open(os.path.join(log_out_path, 'test_Paras.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    trainSet_size = Params['size2train']                   # 训练集大小
    batchSize_train = Params['batch_size2train']           # 训练批量的大小
    batchSize_test = Params['batch_size2test']             # 测试批量的大小
    pt_penalty_init = Params['init_penalty2predict_true']  # 预测值和真值的误差惩罚因子初值,用于处理具有真实值的变量
    wb_penalty = Params['regular_weight']                  # 神经网络参数的惩罚因子
    lr_decay = Params['lr_decay']                          # 学习率额衰减
    init_lr = Params['learning_rate']                      # 初始学习率

    act_func2SIRD = Params['act_Name2SIRD']                 # S, I, Params D 四个神经网络的隐藏层激活函数
    act_func2paras = Params['act_Name2paras']              # 参数网络的隐藏层激活函数

    input_dim = Params['input_dim']                        # 输入维度
    out_dim = Params['output_dim']                         # 输出维度

    flag2S = 'WB2S'
    flag2I = 'WB2I'
    flag2R = 'WB2R'
    flag2D = 'WB2D'
    flag2beta = 'WB2beta'
    flag2gamma = 'WB2gamma'
    flag2mu = 'WB2mu'
    hidden_sird = Params['hidden2SIRD']
    hidden_para = Params['hidden2para']

    if str.upper(Params['model2SIRD']) == 'DNN_FOURIERBASE':
        Weight2S, Bias2S = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2S)
        Weight2I, Bias2I = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2I)
        Weight2R, Bias2R = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2R)
        Weight2D, Bias2D = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2D)
    else:
        Weight2S, Bias2S = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2S)
        Weight2I, Bias2I = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2I)
        Weight2R, Bias2R = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2R)
        Weight2D, Bias2D = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2D)

    if str.upper(Params['model2paras']) == 'DNN_FOURIERBASE':
        Weight2beta, Bias2beta = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2beta)
        Weight2gamma, Bias2gamma = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2gamma)
        Weight2mu, Bias2mu = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2mu)
    else:
        Weight2beta, Bias2beta = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2beta)
        Weight2gamma, Bias2gamma = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2gamma)
        Weight2mu, Bias2mu = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2mu)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (Params['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            T_it = tf.compat.v1.placeholder(tf.float32, name='T_it', shape=[None, out_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[None, out_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[None, out_dim])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[None, out_dim])
            predict_true_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='pt_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            freq2SIRD = Params['freq2SIRD']
            if 'DNN' == str.upper(Params['model2SIRD']):
                SNN_temp = modelUtils.DNN(T_it, Weight2S, Bias2S, hidden_sird, activateIn_name=Params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                INN_temp = modelUtils.DNN(T_it, Weight2I, Bias2I, hidden_sird, activateIn_name=Params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                RNN_temp = modelUtils.DNN(T_it, Weight2R, Bias2R, hidden_sird, activateIn_name=Params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                DNN_temp = modelUtils.DNN(T_it, Weight2D, Bias2D, hidden_sird, activateIn_name=Params['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
            elif 'DNN_SCALE' == str.upper(Params['model2SIRD']):
                SNN_temp = modelUtils.DNN_scale(T_it, Weight2S, Bias2S, hidden_sird, freq2SIRD,
                                              activateIn_name=Params['actIn_Name2SIR'], activate_name=act_func2SIRD)
                INN_temp = modelUtils.DNN_scale(T_it, Weight2I, Bias2I, hidden_sird, freq2SIRD,
                                              activateIn_name=Params['actIn_Name2SIR'], activate_name=act_func2SIRD)
                RNN_temp = modelUtils.DNN_scale(T_it, Weight2R, Bias2R, hidden_sird, freq2SIRD,
                                              activateIn_name=Params['actIn_Name2SIR'], activate_name=act_func2SIRD)
                DNN_temp = modelUtils.DNN_scale(T_it, Weight2D, Bias2D, hidden_sird, freq2SIRD,
                                              activateIn_name=Params['actIn_Name2SIR'], activate_name=act_func2SIRD)
            elif str.upper(Params['model2SIRD']) == 'DNN_FOURIERBASE':
                SNN_temp = modelUtils.DNN_FourierBase(T_it, Weight2S, Bias2S, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                INN_temp = modelUtils.DNN_FourierBase(T_it, Weight2I, Bias2I, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                RNN_temp = modelUtils.DNN_FourierBase(T_it, Weight2R, Bias2R, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                DNN_temp = modelUtils.DNN_FourierBase(T_it, Weight2D, Bias2D, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)

            freq2paras = Params['freq2paras']
            if 'DNN' == str.upper(Params['model2paras']):
                in_beta = modelUtils.DNN(T_it, Weight2beta, Bias2beta, hidden_para,
                                        activateIn_name=Params['actIn_Name2paras'], ctivate_name=act_func2paras)
                in_gamma = modelUtils.DNN(T_it, Weight2gamma, Bias2gamma, hidden_para,
                                        activateIn_name=Params['actIn_Name2paras'], activate_name=act_func2paras)
                in_mu = modelUtils.DNN(T_it, Weight2mu, Bias2mu, hidden_para,
                                        activateIn_name=Params['actIn_Name2paras'], activate_name=act_func2paras)

            elif 'DNN_SCALE' == str.upper(Params['model2paras']):
                in_beta = modelUtils.DNN_scale(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                             activateIn_name=Params['actIn_Name2paras'], activate_name=act_func2paras)
                in_gamma = modelUtils.DNN_scale(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                              activateIn_name=Params['actIn_Name2paras'], activate_name=act_func2paras)
                in_mu = modelUtils.DNN_scale(T_it, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                              activateIn_name=Params['actIn_Name2paras'], activate_name=act_func2paras)                                              
            elif str.upper(Params['model2SIRD']) == 'DNN_FOURIERBASE':
                in_beta = modelUtils.DNN_FourierBase(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                   activate_name=act_func2paras, sFourier=1.0)
                in_gamma = modelUtils.DNN_FourierBase(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                    activate_name=act_func2paras, sFourier=1.0)
                in_mu = modelUtils.DNN_FourierBase(T_it, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                    activate_name=act_func2paras, sFourier=1.0)
            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            # 在归一化条件下: 如果总的“人口”和归一化"人口"的数值一致，这样的话，归一化后的数值会很小
            # beta = tf.square(in_beta)
            beta = tf.nn.sigmoid(in_beta)
            gamma = tf.nn.sigmoid(in_gamma)
            mu = 0.1 * tf.nn.sigmoid(in_mu)

            # S_NN = SNN_temp
            # I_NN = INN_temp
            # R_NN = RNN_temp
            # D_NN = DNN_temp

            # S_NN = tf.nn.relu(SNN_temp)
            # I_NN = tf.nn.relu(INN_temp)
            # R_NN = tf.nn.relu(RNN_temp)
            # D_NN = tf.nn.relu(DNN_temp)

            S_NN = tf.nn.sigmoid(SNN_temp)
            I_NN = tf.nn.sigmoid(INN_temp)
            R_NN = tf.nn.sigmoid(RNN_temp)
            D_NN = tf.nn.sigmoid(DNN_temp)

            # S_NN = tf.tanh(SNN_temp)
            # I_NN = tf.tanh(INN_temp)
            # R_NN = tf.tanh(RNN_temp)
            # D_NN = tf.tanh(DNN_temp)

            # N_NN = S_NN + I_NN + R_NN + D_NN

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dD_NN2t = tf.gradients(D_NN, T_it)[0]
            temp_snn2t = - (beta * S_NN * I_NN) / (S_NN + I_NN)
            temp_inn2t = (beta * S_NN * I_NN) / (S_NN + I_NN) - gamma * I_NN - mu * I_NN
            temp_rnn2t = gamma * I_NN
            temp_dnn2t = mu * I_NN

            if str.lower(Params['loss_function']) == 'l2_loss' and Params['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(I_NN - I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(R_NN - R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(D_NN - D_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(Params['loss_function']) == 'l2_loss' and Params['scale_up'] == 1:
                scale_up = Params['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.square(scale_up*S_NN - scale_up*S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(scale_up*I_NN - scale_up*I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(scale_up*R_NN - scale_up*R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(scale_up*D_NN - scale_up*D_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(Params['loss_function']) == 'lncosh_loss' and Params['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(S_NN - S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(I_NN - I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(R_NN - R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(D_NN - D_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
            elif str.lower(Params['loss_function']) == 'lncosh_loss' and Params['scale_up'] == 1:
                scale_up = Params['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(scale_up*S_NN - scale_up*S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*I_NN - scale_up*I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*R_NN - scale_up*R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*D_NN - scale_up*D_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))

            if Params['regular_weight_model'] == 'L1':
                regular_WB2S = modelUtils.regular_weights_biases_L1(Weight2S, Bias2S)
                regular_WB2I = modelUtils.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = modelUtils.regular_weights_biases_L1(Weight2R, Bias2R)
                regular_WB2D = modelUtils.regular_weights_biases_L1(Weight2D, Bias2D)
                regular_WB2Beta = modelUtils.regular_weights_biases_L1(Weight2beta, Bias2beta)
                regular_WB2Gamma = modelUtils.regular_weights_biases_L1(Weight2gamma, Bias2gamma)
                regular_WB2Mu = modelUtils.regular_weights_biases_L1(Weight2mu, Bias2mu)
            elif Params['regular_weight_model'] == 'L2':
                regular_WB2S = modelUtils.regular_weights_biases_L2(Weight2S, Bias2S)
                regular_WB2I = modelUtils.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = modelUtils.regular_weights_biases_L2(Weight2R, Bias2R)
                regular_WB2D = modelUtils.regular_weights_biases_L2(Weight2D, Bias2D)
                regular_WB2Beta = modelUtils.regular_weights_biases_L2(Weight2beta, Bias2beta)
                regular_WB2Gamma = modelUtils.regular_weights_biases_L2(Weight2gamma, Bias2gamma)
                regular_WB2Mu = modelUtils.regular_weights_biases_L2(Weight2mu, Bias2mu)
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
            Loss2I = predict_true_penalty * LossI_Net_obs + Loss2dI + PWB2I
            # Loss2R = Loss2dR + PWB2R
            Loss2R = predict_true_penalty * LossR_Net_obs + Loss2dR + PWB2R
            # Loss2D = Loss2dD + PWB2D
            Loss2D = predict_true_penalty * LossD_Net_obs + Loss2dD + PWB2D

            Loss = Loss2S + Loss2I + Loss2R + Loss2D + PWB2Beta + PWB2Gamma + PWB2Mu

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if Params['train_model'] == 'train_group':
                train_Loss2S = my_optimizer.minimize(Loss2S, global_step=global_steps)
                train_Loss2I = my_optimizer.minimize(Loss2I, global_step=global_steps)
                train_Loss2R = my_optimizer.minimize(Loss2R, global_step=global_steps)
                train_Loss2D = my_optimizer.minimize(Loss2D, global_step=global_steps)
                train_Losses = tf.group(train_Loss2S, train_Loss2I, train_Loss2R, train_Loss2D)
            elif Params['train_model'] == 'train_union_loss':
                train_Losses = my_optimizer.minimize(Loss, global_step=global_steps)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_all = [], [], [], [], []

    test_epoch = []
    test_mse2I_all, test_rel2I_all = [], []

    filename = 'data/minnesota3.csv'
    date, data2S, data2I, data2R, data2D = DNN_data.load_data(filename, N=3458790)

    # assert (trainSet_size + batchSize_test <= len(data2I))

    # 不归一化数据
    train_data, test_data = DNN_data.split_data2(date, data2S, data2I, data2R, data2D, train_size=0.75)
    # 按顺序取出列表中的数据
    train_date, train_data2s, train_data2i, train_data2r, train_data2d, *_ = train_data
    test_date, test_data2s, test_data2i, test_data2r, test_data2d, *_ = test_data

    nbatch2train = np.ones(batchSize_train, dtype=np.float32) * float(Params['total_population'])

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)

    # 由于将数据拆分为训练数据和测试数据时，进行了归一化处理，故这里不用归一化
    s_obs_test = DNN_data.sample_testData_serially2(test_data2s, batchSize_test)
    i_obs_test = DNN_data.sample_testData_serially2(test_data2i, batchSize_test)
    r_obs_test = DNN_data.sample_testData_serially2(test_data2r, batchSize_test)
    d_obs_test = DNN_data.sample_testData_serially2(test_data2d, batchSize_test)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True) 
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True     # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tmp_lr = init_lr
        for i_epoch in range(Params['max_epoch'] + 1):
            t_batch, s_obs, i_obs, r_obs, d_obs = \
                DNN_data.sample_data(train_date, train_data2s, train_data2i, train_data2r, train_data2d,
                                        batchsize=batchSize_train, normalFactor=1.0, sampling_opt=Params['opt2sample'])
            n_obs = nbatch2train.reshape(batchSize_train, 1)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if Params['activate_stage_penalty'] == 1:
                if i_epoch < int(Params['max_epoch'] / 10):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < int(Params['max_epoch'] / 5):
                    temp_penalty_pt = 10 * pt_penalty_init
                elif i_epoch < int(Params['max_epoch'] / 4):
                    temp_penalty_pt = 50 * pt_penalty_init
                elif i_epoch < int(Params['max_epoch'] / 2):
                    temp_penalty_pt = 100 * pt_penalty_init
                elif i_epoch < int(3 * Params['max_epoch'] / 4):
                    temp_penalty_pt = 200 * pt_penalty_init
                else:
                    temp_penalty_pt = 500 * pt_penalty_init
            elif Params['activate_stage_penalty'] == 2:
                if i_epoch < int(Params['max_epoch'] / 3):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < 2 * int(Params['max_epoch'] / 3):
                    temp_penalty_pt = 10 * pt_penalty_init
                else:
                    temp_penalty_pt = 50 * pt_penalty_init
            else:
                temp_penalty_pt = pt_penalty_init

            _, loss_s, loss_i, loss_r, loss_d, loss, pwb2s, pwb2i, pwb2r, pwb2d = sess.run(
                [train_Losses, Loss2S, Loss2I, Loss2R, Loss2D, Loss, PWB2S, PWB2I, PWB2R, PWB2D],
                feed_dict={T_it: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs, in_learning_rate: tmp_lr,
                           predict_true_penalty: temp_penalty_pt})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            loss_all.append(loss)

            if i_epoch % 1000 == 0:
                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, D_NN, beta, gamma 的训练结果
                s_nn2train, i_nn2train, r_nn2train, d_nn2train = sess.run(
                    [S_NN, I_NN, R_NN, D_NN], feed_dict={T_it: np.reshape(train_date, [-1, 1])})
                beta2train, gamma2train, mu2train = sess.run(
                    [beta, gamma, mu], feed_dict={T_it: np.reshape(train_date, [-1, 1])})

                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的测试结果
                test_epoch.append(i_epoch / 1000)
                s_nn2test, i_nn2test, r_nn2test, d_nn2test, beta_test, gamma_test, mu_test = sess.run(
                    [S_NN, I_NN, R_NN, D_NN, beta, gamma, mu], feed_dict={T_it: test_t_bach})
                point_ERR2I = np.square(i_nn2test - i_obs_test)
                test_mse2I = np.mean(point_ERR2I)
                test_mse2I_all.append(test_mse2I)
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2I_all.append(test_rel2I)

                # --------以下代码为输出训练过程中 S_NN_temp, I_NN_temp, R_NN_temp, in_beta, in_gamma 的测试结果-------------
                s_nn_temp2test, i_nn_temp2test, r_nn_temp2test, d_nn_temp2test, in_beta_test, in_gamma_test, in_mu_test = sess.run(
                    [SNN_temp, INN_temp, RNN_temp, DNN_temp, in_beta, in_gamma, in_mu],
                    feed_dict={T_it: test_t_bach})


        # 把groud truth 和 predict数据存储在csv文件里
        data_dic = {'train_data2s': train_data2s,'s_nn2train': np.squeeze(s_nn2train),
                    'train_data2i': train_data2i,'i_nn2train': np.squeeze(i_nn2train),
                    'train_data2r': train_data2r,'r_nn2train': np.squeeze(r_nn2train),
                    'train_data2d': train_data2d,'d_nn2train': np.squeeze(d_nn2train)
        }
        data_df = pd.DataFrame.from_dict(data_dic)
        data_df.to_csv(Params['FolderName'] + '/sird_results.csv', index = False)

        paras_dic = {'beta2train': np.squeeze(beta2train), 
                     'gamma2train': np.squeeze(gamma2train), 
                     'mu2train': np.squeeze(mu2train)}
        paras_df = pd.DataFrame.from_dict(paras_dic)
        paras_df.to_csv(Params['FolderName'] + '/paras_results.csv', index = False)

        # save loss data
        loss_dic = {'loss_s':loss_s_all,
                    'loss_i':loss_i_all,
                    'loss_r':loss_r_all,
                    'loss_d':loss_d_all}
        loss_df = pd.DataFrame.from_dict(loss_dic)
        loss_df.to_csv(Params['FolderName'] + '/loss_results.csv', index = False)

if __name__ == "__main__":
    Params = {}
    Params['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SIRD2covid'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    Params['seed'] = 43
    timeFolder = datetime.now().strftime("%Y%m%d_%H%M")       # 当前时间为文件夹名
    Params['FolderName'] = os.path.join(OUT_DIR, timeFolder)  # 路径连接
    FolderName = Params['FolderName']
    if not os.path.exists(FolderName):
        os.makedirs(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    Params['max_epoch'] = 200000 #200000

    # ----------------------------------------- Covid 设置 ---------------------------------
    Params['eqs_name'] = 'SIRD'
    Params['input_dim'] = 1                       # 输入维数，即问题的维数(几元问题)
    Params['output_dim'] = 1                      # 输出维数
    Params['total_population'] = 3000000 #9776000          # 总的“人口”数量

    # ------------------------------------  神经网络的设置  ----------------------------------------
    Params['size2train'] = 70                    # 训练集的大小
    Params['batch_size2train'] = 20              # 训练数据的批大小
    Params['batch_size2test'] = 10               # 训练数据的批大小
    # 训练集的选取方式
    # 1. 随机采样'random_sample'
    # 2.随机采样后按时间排序'rand_sample_sort'
    # 3.随机窗口采样(以随机点为基准，然后滑动窗口采样)'windows_rand_sample'
    Params['opt2sample'] = 'rand_sample_sort'

    # loss震荡幅度过大，处理方式
    # 1.不阶段惩罚ground truth 和predicted的差值 init_penalty2predict_true尽量大一些（50，100， 1000），
    # 2. activate_stage_penalty 设置为 0.（为1时是阶段惩罚）
    Params['init_penalty2predict_true'] = 50     # Regularization parameter for boundary conditions
    Params['activate_stage_penalty'] = 1         # 是否开启阶段调整惩罚项，0 代表不调整，非 0 代表调整
    # [1, 20, 50, 100, 1000]
    if Params['activate_stage_penalty'] == 1 or Params['activate_stage_penalty'] == 2:
        Params['init_penalty2predict_true'] = 1

    Params['regular_weight_model'] = 'L2'    # The model of regular weights and biases 'L0''L1''L2'
    Params['regular_weight'] = 0.00005  # Regularization parameter for weights0.00001, 0.00005, 0.0001, 0.0005, 0.001

    Params['optimizer_name'] = 'Adam'              # 优化器
    Params['loss_function'] = 'L2_loss'            # 损失函数的类型 'L2_loss'  'lncosh_loss'
    Params['scale_up'] = 1                         # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    Params['scale_factor'] = 100                   # scale_factor 用来对数值进行尺度提升，如1e-6量级提升到1e-2量级

    # Params['train_model'] = 'train_group'        # 训练模式:各个不同的loss捆绑打包训练
    Params['train_model'] = 'train_union_loss'     # 训练模式:各个不同的loss累加在一起，训练

    if 50000 < Params['max_epoch']:
        Params['learning_rate'] = 2e-3             # 学习率 2e-4
        Params['lr_decay'] = 1e-4                  # 学习率 decay 5e-5
    elif (20000 < Params['max_epoch'] and 50000 >= Params['max_epoch']):
        Params['learning_rate'] = 1e-4             # 学习率1e-3, 2e-4, 1e-4
        Params['lr_decay'] = 5e-5                  # 学习率 decay 1e-4 
    else:
        Params['learning_rate'] = 5e-5             # 学习率
        Params['lr_decay'] = 1e-5                  # 学习率 decay

    # SIRD和参数网络模型的选择 ['DNN', 'DNN_scale', 'DNN_scaleOut', 'DNN_FourierBase']
    Params['model2SIRD'] =  'DNN_FourierBase'
    Params['model2paras'] = 'DNN_FourierBase'

    # SIRD和参数网络模型的隐藏层单元数目
    if Params['model2SIRD'] == 'DNN_FourierBase':
        Params['hidden2SIRD'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # [(70, 50, 30, 30, 20), (80, 80, 60, 40, 40, 20), (100, 100, 80, 60, 60, 40), (200, 100, 100, 80, 50, 50)]
        Params['hidden2SIRD'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570

    if Params['model2paras'] == 'DNN_FourierBase':
        Params['hidden2para'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # [(70, 50, 30, 30, 20), (80, 80, 60, 40, 40, 20), (100, 100, 80, 60, 60, 40), (200, 100, 100, 80, 50, 50)]
        Params['hidden2para'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570

    # SIRD和参数网络模型的尺度因子
    if Params['model2SIRD'] != 'DNN':
        Params['freq2SIRD'] = np.concatenate(([1], np.arange(1, 20)), axis=0)
    if Params['model2paras'] != 'DNN':
        Params['freq2paras'] = np.concatenate(([1], np.arange(1, 20)), axis=0)

    # SIRD和参数网络模型为傅里叶网络和尺度网络时，重复高频因子或者低频因子
    if Params['model2SIRD'] == 'DNN_FourierBase' or Params['model2SIRD'] == 'DNN_scale':
        Params['if_repeat_High_freq2SIRD'] = False
    if Params['model2paras'] == 'DNN_FourierBase' or Params['model2paras'] == 'DNN_scale':
        Params['if_repeat_High_freq2paras'] = False

    # SIRD和参数网络模型的激活函数的选择
    # ['relu', 'leaky_relu', 'sigmod', 'tanh','srelu', 'sin', 'sinAddcos', 'elu', 'gelu', 'mgelu', 'linear']
    Params['actIn_Name2SIRD'] = 'tanh'
    Params['act_Name2SIRD'] = 'tanh'  # 这个激活函数比较s2ReLU合适
    Params['actIn_Name2paras'] = 'tanh'
    Params['act_Name2paras'] = 'tanh'  # 这个激活函数比较s2ReLU合适

    solve_SIRD2COVID(Params)

