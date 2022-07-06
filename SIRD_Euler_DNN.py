import os
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import modelUtils
import DNN_tools
import DNN_data
import plotData
import saveData

import dataUtils
import modelUtils
from datetime import datetime
import pandas as pd

import argparse

# tf2兼容tf1
tf.compat.v1.disable_eager_execution()

# 在模型中除以N

DECAY_STEPS = 50000
DECAY_RATE = 0.8

parser = argparse.ArgumentParser()

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--data_fname', type=str, default='./data/minnesota3.csv',
                    help='data path')

parser.add_argument('--train_epoches', type=int, default=200000,
                    help='max epoch during training.')

parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer')

parser.add_argument('--epochs_per_eval', type=int, default=10000,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--eqs_name', type=str, default='SIRD',
                    help='Whether to use debugger to track down bad values during training.')

parser.add_argument('--batach_size', type=int, default=16,
                    help='.')

parser.add_argument('--sample_method', type=str, default='sequential_sort',
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


def solve_SIRD2COVID(params):
    if not os.path.exists(params['FolderName']):  # 判断路径是否存在
        os.mkdir(params['FolderName']) 
    
    # 记录训练日志
    log_fileout = open(os.path.join(params['FolderName'], 'log_train.txt'), 'w')
    dataUtils.dictionary_out2file2(params, log_fileout)

    # trainSet_szie = params['size2train']                   # 训练集大小
    batchSize_train = params['batch_size2train']           # 训练批量的大小
    # batchSize_test = params['batch_size2test']             # 测试批量的大小


    # 创建矩阵
    AI = tf.eye(batchSize_train, dtype=tf.float32) * 2
    Ones_mat = tf.ones([batchSize_train, batchSize_train], dtype=tf.float32)
    A_diag = tf.linalg.band_part(Ones_mat, 0, 1)
    Amat = AI - A_diag

    # 初始化神经网络参数
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
            T_train = tf.compat.v1.placeholder(tf.float32, name='T_train', shape=[batchSize_train, params['output_dim']])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[batchSize_train, params['output_dim']])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[batchSize_train, params['output_dim']])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[batchSize_train, params['output_dim']])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[batchSize_train, params['output_dim']])

            T_test = tf.compat.v1.placeholder(tf.float32, name='T_test', shape=[None, params['output_dim']])

            predict_true_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='pt_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            freq2paras = params['freq2paras']
            with tf.name_scope('paras_network'):
                if 'DNN' == str.upper(params['params_network']):
                    in_beta2train = modelUtils.DNN(T_train, Weight2beta, Bias2beta, params['hidden_params'],
                                            activateIn_name=params['activateIn_params'], ctivate_name=params['activate_params'])
                    in_gamma2train = modelUtils.DNN(T_train, Weight2gamma, Bias2gamma, params['hidden_params'],
                                            activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])
                    in_mu2train = modelUtils.DNN(T_train, Weight2mu, Bias2mu, params['hidden_params'],
                                            activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])

                elif 'DNN_SCALE' == str.upper(params['params_network']):
                    in_beta2train = modelUtils.DNN_scale(T_train, Weight2beta, Bias2beta, params['hidden_params'], freq2paras,
                                                activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])
                    in_gamma2train = modelUtils.DNN_scale(T_train, Weight2gamma, Bias2gamma, params['hidden_params'], freq2paras,
                                                activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])
                    in_mu2train = modelUtils.DNN_scale(T_train, Weight2mu, Bias2mu, params['hidden_params'], freq2paras,
                                                activateIn_name=params['activateIn_params'], activate_name=params['activate_params'])                                              
                elif str.upper(params['sird_network']) == 'DNN_FOURIERBASE':
                    in_beta2train = modelUtils.DNN_FourierBase(T_train, Weight2beta, Bias2beta, params['hidden_params'], freq2paras,
                                                    activate_name=params['activate_params'], sFourier=1.0)
                    in_gamma2train = modelUtils.DNN_FourierBase(T_train, Weight2gamma, Bias2gamma, params['hidden_params'], freq2paras,
                                                        activate_name=params['activate_params'], sFourier=1.0)
                    in_mu2train = modelUtils.DNN_FourierBase(T_train, Weight2mu, Bias2mu, params['hidden_params'], freq2paras,
                                                        activate_name=params['activate_params'], sFourier=1.0)

            # 激活函数
            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            # 在归一化条件下: 如果总的“人口”和归一化"人口"的数值一致，这样的话，归一化后的数值会很小
            # tf.square(), tf.tanh(), tf.nn.relu(), tf.abs(), modelUtils.gauss()
            betaNN2train = tf.nn.relu(in_beta2train)
            # betaNN2train  = tf.nn.sigmoid(in_beta2train)
            gammaNN2train = tf.nn.sigmoid(in_gamma2train)
            muNN2train = 0.1 * tf.nn.sigmoid(in_mu2train)

            # Euler
            dS2dt = tf.matmul(Amat[0:-1, :], S_observe)
            dI2dt = tf.matmul(Amat[0:-1, :], I_observe)
            dR2dt = tf.matmul(Amat[0:-1, :], R_observe)
            dD2dt = tf.matmul(Amat[0:-1, :], D_observe)

            temp_s2t = -betaNN2train[0:-1, 0] * S_observe[0:-1, 0] * I_observe[0:-1, 0]/(S_observe[0:-1, 0] + I_observe[0:-1, 0])
            temp_i2t = betaNN2train[0:-1, 0] * S_observe[0:-1, 0] * I_observe[0:-1, 0]/(S_observe[0:-1, 0] + I_observe[0:-1, 0]) - \
                       gammaNN2train[0:-1, 0] * I_observe[0:-1, 0] - muNN2train[0:-1, 0] * I_observe[0:-1, 0]
            temp_r2t = gammaNN2train[0:-1, 0] * I_observe[0:-1, 0]
            temp_d2t = muNN2train[0:-1, 0] * I_observe[0:-1, 0]

            if params['loss_function'].lower() == 'l2_loss':
                Loss2dS = tf.reduce_mean(tf.square(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1])))
                Loss2dI = tf.reduce_mean(tf.square(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1])))
                Loss2dR = tf.reduce_mean(tf.square(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1])))
                Loss2dD = tf.reduce_mean(tf.square(dD2dt - tf.reshape(temp_d2t, shape=[-1, 1])))
            elif params['loss_function'].lower() == 'lncosh_loss':
                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1]))))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1]))))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1]))))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD2dt - tf.reshape(temp_d2t, shape=[-1, 1]))))
            
            # 正则化
            regular_func = lambda a, b: tf.constant(0.0)
            if params['regular_method'] == 'L1':
                regular_func = modelUtils.regular_weights_biases_L1
            elif params['regular_method'] == 'L2':
                regular_func = modelUtils.regular_weights_biases_L2
            regular_WB2Beta = regular_func(Weight2beta, Bias2beta)
            regular_WB2Gamma = regular_func(Weight2gamma, Bias2gamma)
            regular_WB2Mu = regular_func(Weight2mu, Bias2mu) 

            PWB2Beta = params['regular_weight'] * regular_WB2Beta
            PWB2Gamma = params['regular_weight'] * regular_WB2Gamma
            PWB2Mu = params['regular_weight'] * regular_WB2Mu

            Loss = Loss2dS + Loss2dI + Loss2dR + Loss2dD + PWB2Beta + PWB2Gamma + PWB2Mu

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            train_Losses = my_optimizer.minimize(Loss, global_step=global_steps)

    loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_all = [], [], [], [], []

    test_epoch = []
    test_mse2S_all, test_rel2S_all = list(), list()
    test_mse2I_all, test_rel2I_all = list(), list()
    test_mse2R_all, test_rel2R_all = list(), list()
    test_mse2D_all, test_rel2D_all = list(), list()

    data_list = dataUtils.load_data(params['data_fname'], N=3450000)
    date, data2S, data2I, data2R, data2D, *_ = data_list

    # 是否归一化数据，取决于normalFactor的值
    train_data, test_data = DNN_data.split_data(date, data2S, data2I, data2R, data2D, train_size=1.0, normalFactor=params['normalize_population'])
    # 按顺序取出列表中的数据
    train_date, train_data2s, train_data2i, train_data2r, train_data2d, *_ = train_data
    # test_date, test_data2s, test_data2i, test_data2r, test_data2d, *_ = test_data
    test_date, test_data2s, test_data2i, test_data2r, test_data2d, *_ = train_data  

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    batchSize_test = len(train_data)
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True                        # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                            # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i_epoch in range(params['max_epoch'] + 1):
            t_batch, s_obs, i_obs, r_obs, d_obs = \
                DNN_data.sample_data(train_date, train_data2s, train_data2i, train_data2r, train_data2d,
                                        window_size=7, sampling_opt=params['opt2sample'])
            _, loss_s, loss_i, loss_r, loss_d, loss, pwb2beta, pwb2gamma, pwb2mu = sess.run(
                [train_Losses, Loss2dS, Loss2dI, Loss2dR, Loss2dD, Loss, PWB2Beta, PWB2Gamma, PWB2Mu],
                feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs, in_learning_rate: tmp_lr})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            loss_all.append(loss)

            if i_epoch % 1000 == 0:
                print(i_epoch)
                # 以下代码为输出训练过程中 beta, gamma mu的训练结果
                test_beta, test_gamma, test_mu = sess.run([betaNN2test, gammaNN2test, muNN2test], 
                                                          feed_dict={T_test: np.reshape(test_date, [-1, 1])})
                # 以下代码为输出训练过程中 in_beta, in_gamma, in_mu 的测试结果
                in_test_beta, in_test_gamma, in_test_mu = sess.run([in_beta2test, in_gamma2test, in_mu2test],
                                                                   feed_dict={T_test: test_t_batch})

        # 把groud truth 和 predict数据存储在csv文件里
        # data_dic = {'train_data2s': train_data2s,'s_nn2train': np.squeeze(temp_s2t),
        #             'train_data2i': train_data2i,'i_nn2train': np.squeeze(temp_i2t),
        #             'train_data2r': train_data2r,'r_nn2train': np.squeeze(temp_r2t),
        #             'train_data2d': train_data2d,'d_nn2train': np.squeeze(temp_d2t)
        # }
        # data_df = pd.DataFrame.from_dict(data_dic)
        # data_df.to_csv(params['FolderName'] + '/sird_results.csv', index = False)
        # beta2train, gamma2train, mu2train = test_beta, test_gamma, test_mu
        paras_dic = {'beta2train': np.squeeze(test_beta), 
                     'gamma2train': np.squeeze(test_gamma), 
                     'mu2train': np.squeeze(test_mu)}
        paras_df = pd.DataFrame.from_dict(paras_dic)
        paras_df.to_csv(params['FolderName'] + '/paras_results.csv', index = False)

        # save loss data
        loss_dic = {'loss_s':loss_s_all,
                    'loss_i':loss_i_all,
                    'loss_r':loss_r_all,
                    'loss_d':loss_d_all}
        loss_df = pd.DataFrame.from_dict(loss_dic)
        loss_df.to_csv(params['FolderName'] + '/loss_results.csv', index = False)

def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    if FLAGS.clean_model_dir:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

    params={'data_fname': FLAGS.data_fname,
            'optimizer': FLAGS.optimizer,
            'train_epoches': FLAGS.train_epoches,
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
            'total_population': 3450000
    }

    params['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SIRD2covid'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
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
    if (params['activate_stage_penalty'] == True):
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

    solve_SIRD2COVID(params)

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)