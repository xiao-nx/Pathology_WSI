import numpy as np
import tensorflow as tf

# tf2兼容tf1
tf.compat.v1.disable_eager_execution()


def Xavier_init_NN_Fourier(in_size, out_size, hidden_layers, Flag='flag', varcoe=0.5):
    with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []   # 权重列表，存储隐藏层权重
        Biases = []    # 偏置列表，存储隐藏层偏置
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.compat.v1.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)

        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            if 0 == i_layer:
                W = tf.compat.v1.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer]*2, hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.compat.v1.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                              initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            else:
                W = tf.compat.v1.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.compat.v1.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                              initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.compat.v1.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def Xavier_init_NN(in_size, out_size, hidden_layers, Flag='flag', varcoe=0.5):
    with tf.compat.v1.variable_scope('WB_scope', reuse=tf.compat.v1.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []  # 权重列表，存储隐藏层权重
        Biases = []   # 偏置列表，存储隐藏层偏置
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.compat.v1.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                      dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.compat.v1.get_variable(
                name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            B = tf.compat.v1.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                          initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置,将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.compat.v1.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.compat.v1.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                                      initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases

# ----------------------------------- 正则化 -----------------------------------------------
def regular_weights_biases_L1(weights, biases):
    # L1正则化权重和偏置
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.abs(weights[i_layer1]), keepdims=False)
        regular_b = regular_b + tf.reduce_sum(tf.abs(biases[i_layer1]), keepdims=False)
    return regular_w + regular_b


# L2正则化权重和偏置
def regular_weights_biases_L2(weights, biases):
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.square(weights[i_layer1]), keepdims=False)
        regular_b = regular_b + tf.reduce_sum(tf.square(biases[i_layer1]), keepdims=False)
    return regular_w + regular_b


# ---------------------------------------------- my activations -----------------------------------------------
def linear(x):
    return x


def mysin(x):
    # return tf.sin(2*np.pi*x)
    return tf.sin(x)
    # return 0.5*tf.sin(x)


def srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)


def s2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)
    # return 1.5*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)
    # return 1.25*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)


def sinAddcos(x):
    return 0.5*(tf.sin(x) + tf.cos(x))
    # return tf.sin(x) + tf.cos(x)


def sinAddcos_sReLu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*(tf.sin(2*np.pi*x) + tf.cos(2*np.pi*x))


def s3relu(x):
    # return 0.5*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
    # return 0.21*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
    # return tf.nn.relu(1 - x) * tf.nn.relu(x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x))   # (work不好)
    # return tf.nn.relu(1 - x) * tf.nn.relu(1 + x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x)) #（不work）
    return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*tf.abs(x))      # work 不如 s2relu
    # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*x)            # work 不如 s2relu
    # return 1.5*tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(np.pi*x)
    # return tf.nn.relu(1 - x) * tf.nn.relu(x+0.5) * tf.sin(2 * np.pi * x)


def csrelu(x):
    # return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.cos(np.pi*x)
    return 1.5*tf.nn.relu(1 - x) * tf.nn.relu(x) * tf.cos(np.pi * x)
    # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.cos(np.pi*x)


def stanh(x):
    return tf.tanh(x)*tf.sin(2*np.pi*x)


def gauss(x):
    return tf.exp(-1.0 * x * x)
    # return 0.2*tf.exp(-4*x*x)
    # return 0.25*tf.exp(-7.5*(x-0.5)*(x-0.5))


def mexican(x):
    return (1-x*x)*tf.exp(-0.5*x*x)


def modify_mexican(x):
    # return 1.25*x*tf.exp(-0.25*x*x)
    # return x * tf.exp(-0.125 * x * x)
    return x * tf.exp(-0.075*x * x)
    # return -1.25*x*tf.exp(-0.25*x*x)


def sm_mexican(x):
    # return tf.sin(np.pi*x) * x * tf.exp(-0.075*x * x)
    # return tf.sin(np.pi*x) * x * tf.exp(-0.125*x * x)
    return 2.0*tf.sin(np.pi*x) * x * tf.exp(-0.5*x * x)


def singauss(x):
    # return 0.6 * tf.exp(-4 * x * x) * tf.sin(np.pi * x)
    # return 0.6 * tf.exp(-5 * x * x) * tf.sin(np.pi * x)
    # return 0.75*tf.exp(-5*x*x)*tf.sin(2*np.pi*x)
    # return tf.exp(-(x-0.5) * (x - 0.5)) * tf.sin(np.pi * x)
    # return 0.25 * tf.exp(-3.5 * x * x) * tf.sin(2 * np.pi * x)
    # return 0.225*tf.exp(-2.5 * (x - 0.5) * (x - 0.5)) * tf.sin(2*np.pi * x)
    return 0.225 * tf.exp(-2 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
    # return 0.4 * tf.exp(-10 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
    # return 0.45 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(np.pi * x)
    # return 0.3 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(2 * np.pi * x)
    # return tf.sin(2*np.pi*tf.exp(-0.5*x*x))


def powsin_srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)*tf.sin(2*np.pi*x)


def sin2_srelu(x):
    return 2.0*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(4*np.pi*x)*tf.sin(2*np.pi*x)


def slrelu(x):
    return tf.nn.leaky_relu(1-x)*tf.nn.leaky_relu(x)


def pow2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.nn.relu(x)


def selu(x):
    return tf.nn.elu(1-x)*tf.nn.elu(x)


def wave(x):
    return tf.nn.relu(x) - 2*tf.nn.relu(x-1/4) + \
           2*tf.nn.relu(x-3/4) - tf.nn.relu(x-1)


def phi(x):
    return tf.nn.relu(x) * tf.nn.relu(x)-3*tf.nn.relu(x-1)*tf.nn.relu(x-1) + 3*tf.nn.relu(x-2)*tf.nn.relu(x-2) \
           - tf.nn.relu(x-3)*tf.nn.relu(x-3)*tf.nn.relu(x-3)


def gelu(x):
    out = x*tf.exp(x)/(1+tf.exp(x))
    return out


def mgelu(x):
    temp2x = np.sqrt(2 / np.pi) * (x + 0.044715 * x * x * x)
    # out = 0.5*+ 0.5*x*tf.tanh(temp2x)
    out = 0.25 * x * tf.tanh(temp2x)
    return out



def DNN(variable_input, Weights, Biases, hiddens, activateIn_name='tanh', activate_name='tanh', activateOut_name='linear'):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
    return:
        output data, dim:NxD', generally D'=1
    """
    activate_dic = {'relu': tf.nn.relu,
                    'leaky_relu': tf.nn.leaky_relu(0.2),
                    'srelu': srelu,
                    'elu': tf.nn.elu,
                    'sin': mysin,
                    'sinaddcos': sinAddcos,
                    'tanh': tf.tanh,
                    'gauss': gauss,
                    'softplus': tf.nn.softplus,
                    'sigmoid': tf.nn.sigmoid,
                    'gelu': gelu,
                    'mgelu': mgelu,
                    'linear': linear
                    }
    assert activateIn_name.lower() in activate_dic
    assert activate_name.lower() in activate_dic
    assert activateOut_name.lower() in activate_dic

    layers = len(hiddens) + 1               # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    hidden_record = 0
    for k in range(layers-1):
        H_pre = H
        W = Weights[k]
        B = Biases[k]
        if k == 0:
            act_in = activate_dic[activateIn_name]
            H = act_in(tf.add(tf.matmul(H, W), B))
        else:
            act_func = activate_dic[activate_name]
            H = act_func(tf.add(tf.matmul(H, W), B))
        if hiddens[k] == hidden_record:
            H = H+H_pre
        hidden_record = hiddens[k]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    act_out = activate_dic[activateOut_name]
    output = act_out(output)
    return output

def DNN_scale(variable_input, Weights, Biases, hiddens, freq_frag, activateIn_name='tanh', activate_name='tanh',
              activateOut_name='linear', repeat_Highfreq=True):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        freq_frag: a list or tuple for scale-factor
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
        repeat_Highfreq: repeat the high-freq factor or not
    return:
        output data, dim:NxD', generally D'=1
    """
    activate_dic = {'relu': tf.nn.relu,
                    'leaky_relu': tf.nn.leaky_relu(0.2),
                    'srelu': srelu,
                    'elu': tf.nn.elu,
                    'sin': mysin,
                    'sinaddcos': sinAddcos,
                    'tanh': tf.tanh,
                    'gauss': gauss,
                    'softplus': tf.nn.softplus,
                    'sigmoid': tf.nn.sigmoid,
                    'gelu': gelu,
                    'mgelu': mgelu,
                    'linear': linear
                    }
    assert activateIn_name.lower() in activate_dic
    assert activate_name.lower() in activate_dic
    assert activateOut_name.lower() in activate_dic

    Unit_num = int(hiddens[0] / len(freq_frag))

    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    mixcoe = np.repeat(freq_frag, Unit_num)

    # 这个的作用是什么？
    if repeat_Highfreq==True:
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        mixcoe = np.concatenate((np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0], mixcoe))

    mixcoe = mixcoe.astype(np.float32)

    layers = len(hiddens) + 1  # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    W_in = Weights[0]
    B_in = Biases[0]
    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)

    act_in = activate_dic[activateIn_name]
    H = act_in(H)

    hidden_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        act_func = activate_dic[activate_name]
        H = act_func(tf.add(tf.matmul(H, W), B))
        if hiddens[k+1] == hidden_record:
            H = H + H_pre
        hidden_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    act_out = activate_dic[activateOut_name]
    output = act_out(output)
    return output



def DNN_adapt_scale(variable_input, Weights, Biases, hiddens, freq_frag, activateIn_name='tanh', activate_name='tanh',
                    activateOut_name='linear', repeat_Highfreq=True):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        freq_frag: a list or tuple for scale-factor
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
        repeat_Highfreq: repeat the high-freq factor or not
    return:
        output data, dim:NxD', generally D'=1
    """
    activate_dic = {'relu': tf.nn.relu,
                    'leaky_relu': tf.nn.leaky_relu(0.2),
                    'srelu': srelu,
                    'elu': tf.nn.elu,
                    'sin': mysin,
                    'sinaddcos': sinAddcos,
                    'tanh': tf.tanh,
                    'gauss': gauss,
                    'softplus': tf.nn.softplus,
                    'sigmoid': tf.nn.sigmoid,
                    'gelu': gelu,
                    'mgelu': mgelu,
                    'linear': linear
                    }
    assert activateIn_name.lower() in activate_dic
    assert activate_name.lower() in activate_dic
    assert activateOut_name.lower() in activate_dic

    Unit_num = int(hiddens[0] / len(freq_frag))

    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    init_mixcoe = np.repeat(freq_frag, Unit_num)

    # 这个的作用是什么？
    if repeat_Highfreq==True:
        init_mixcoe = np.concatenate((init_mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        init_mixcoe = np.concatenate((init_mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0]))

    # 将 int 型的 mixcoe 转化为 发np.flost32 型的 mixcoe，mixcoe[: units[1]]省略了行的维度
    init_mixcoe = init_mixcoe.astype(np.float32)

    layers = len(hiddens)+1                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    W_in = Weights[0]
    B_in = Biases[0]
    mixcoe = tf.get_variable(name='M0', initializer=init_mixcoe)
    # mixcoe = tf.exp(mixcoe)

    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)

    act_in = activate_dic[activateIn_name]
    H = act_in(H)

    hidden_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        act_func = activate_dic[activate_name]
        H = act_func(tf.add(tf.matmul(H, W), B))
        if hiddens[k+1] == hidden_record:
            H = H + H_pre
        hidden_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    act_out = activate_dic[activateOut_name]
    output = act_out(output)
    return output


# FourierBase 代表 cos concatenate sin according to row（i.e. the number of sampling points）
def DNN_FourierBase(variable_input, Weights, Biases, hiddens, freq_frag, activate_name='tanh', activateOut_name='linear',
                    repeat_Highfreq=True, sFourier=1.0):
    """
    Args:
        variable_input: the input data, dim：NxD
        Weights: the weight for each hidden layer
        Biases: the bias for each hidden layer
        hiddens: a list or tuple for hidden-layer, it contains the num of neural units
        freq_frag: a list or tuple for scale-factor
        activateIn_name: the name of activation function for input-layer
        activate_name: the name of activation function for hidden-layer
        activateOut_name: the name of activation function for output-layer
        repeat_Highfreq: repeat the high-freq factor or not
        sFourier：a scale factor for adjust the range of input-layer
    return:
        output data, dim:NxD', generally D'=1
    """

    activate_dic = {'relu': tf.nn.relu,
                    'leaky_relu': tf.nn.leaky_relu(0.2),
                    'srelu': srelu,
                    'elu': tf.nn.elu,
                    'sin': mysin,
                    'sinaddcos': sinAddcos,
                    'tanh': tf.tanh,
                    'gauss': gauss,
                    'softplus': tf.nn.softplus,
                    'sigmoid': tf.nn.sigmoid,
                    'gelu': gelu,
                    'mgelu': mgelu,
                    'linear': linear
                    }
    assert activate_name.lower() in activate_dic
    assert activateOut_name.lower() in activate_dic

    layers = len(hiddens) + 1                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层

    # 计算第一个隐藏单元和尺度标记的比例
    Unit_num = int(hiddens[0] / len(freq_frag))

    # 然后，频率标记按按照比例复制
    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    mixcoe = np.repeat(freq_frag, Unit_num)

    if repeat_Highfreq == True:
        # 如果第一个隐藏单元的长度大于复制后的频率标记，那就按照最大的频率在最后补齐
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0]))

    mixcoe = mixcoe.astype(np.float32)

    W_in = Weights[0]
    B_in = Biases[0]
    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        # H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)
        H = tf.matmul(H, W_in) * mixcoe

    H = sFourier * (tf.concat([tf.cos(H), tf.sin(H)], axis=-1))
    # H = sfactor * (tf.concat([tf.cos(np.pi * H), tf.sin(np.pi * H)], axis=-1))
    # H = sfactor * tf.concat([tf.cos(2 * np.pi * H), tf.sin(2 * np.pi * H)], axis=-1)

    hiddens_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        act_func = activate_dic[activate_name]
        H = act_func(tf.add(tf.matmul(H, W), B))
        if (hiddens[k+1] == hiddens_record) and (k != 0):
            H = H + H_pre
        hiddens_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    act_out = activate_dic[activateOut_name]
    output = act_out(output)
    return output


## 
def dnn_SIRD(date, weights, biases, hidden_layers, activate_func,activateIn_func):
    S_NN = DNN(date, weights, biases, hidden_layers, activate_func,activateIn_func)
    I_NN = DNN(date, weights, biases, hidden_layers, activate_func,activateIn_func)
    R_NN = DNN(date, weights, biases, hidden_layers, activate_func,activateIn_func)
    D_NN = DNN(date, weights, biases, hidden_layers, activate_func,activateIn_func)
    return S_NN, I_NN, R_NN, D_NN

def dnn_params(date, weights, biases, hidden_layers, activate_func,activateIn_func):
    beta = DNN(date, weights, biases, hidden_layers, activate_func,activateIn_func)
    gamma = DNN(date, weights, biases, hidden_layers, activate_func,activateIn_func)
    mu = DNN(date, weights, biases, hidden_layers, activate_func,activateIn_func)

    return beta, gamma, mu 

def dnnFourier_SIRD(date, weights, biases, hidden_layers, activate_func,activateIn_func,sFourier=1.0):
    S_NN = DNN_FourierBase(date, weights, biases, hidden_layers, activate_func,activateIn_func,sFourier=1.0)
    I_NN = DNN_FourierBase(date, weights, biases, hidden_layers, activate_func,activateIn_func,sFourier=1.0)
    R_NN = DNN_FourierBase(date, weights, biases, hidden_layers, activate_func,activateIn_func,sFourier=1.0)
    D_NN = DNN_FourierBase(date, weights, biases, hidden_layers, activate_func,activateIn_func,sFourier=1.0)

    return S_NN, I_NN, R_NN, D_NN