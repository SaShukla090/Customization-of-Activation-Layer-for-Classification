#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.training import moving_averages


def variable_weight_bn(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)


def batch_norm(x, decay=0.999, epsilon=1e-03, scope="scope"):
    x_shape = x.get_shape()
    input_channels = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))

    with tf.variable_scope(scope):
        beta = variable_weight_bn("beta", [input_channels, ],
                               initializer=tf.zeros_initializer())
        gamma = variable_weight_bn("gamma", [input_channels, ],
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = variable_weight_bn("moving_mean", [input_channels, ],
                                      initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = variable_weight_bn("moving_variance", [input_channels],
                                          initializer=tf.ones_initializer(), trainable=False)

    mean, variance = tf.nn.moments(x, axes=reduce_dims)
    update_move_mean = moving_averages.assign_moving_average(moving_mean, mean, decay=decay)
    update_move_variance = moving_averages.assign_moving_average(moving_variance, variance, decay=decay)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)





def AdaptiveFunc(z, activation = "relu"):
    if activation == "relu":
        # with tf.variable_scope(name) as scope:
            # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffR = tf.nn.relu(z)
        return ffR 

    elif activation == "tanh":
        # with tf.variable_scope(name) as scope:
            # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffT = tf.tanh(z)
        return ffT

    elif activation == "sigmoid":
        # with tf.variable_scope(name) as scope:
        z = batch_norm(z)
        ffS = tf.sigmoid(z)
        return ffS

    elif activation == "arelu":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
            # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        return ffR

    elif activation == "atanh":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
            # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        return ffT

    elif activation == "asigmoid":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
        # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        return ffS

    elif activation == "m1":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
        alpha = tf.Variable(tf.constant(1.0), name="alpha")
        beta = tf.Variable(tf.constant(1.0), name="beta")
        gamma = tf.Variable(tf.constant(1.0), name="gamma")
            # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        ff = (alpha/(alpha + beta + gamma))*ffT + (beta/(alpha + beta + gamma))*ffS + (gamma/(alpha + beta + gamma))*ffR #m1
        return ff

    elif activation == "m2":
        # with tf.variable_scope(name) as scope:
        alpha = tf.Variable(tf.constant(1.0), name="alpha")
        beta = tf.Variable(tf.constant(1.0), name="beta")
        gamma = tf.Variable(tf.constant(1.0), name="gamma")
            # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffS = tf.sigmoid(z)
        ffT = tf.tanh(z)
        ffR = tf.nn.relu(z)
        ff = (alpha/(alpha + beta + gamma))*ffT + (beta/(alpha + beta + gamma))*ffS + (gamma/(alpha + beta + gamma))*ffR #m1
        return  ff

    elif activation == "onem1":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
        # z = batch_norm(z, scope="bn" + name)
        z = batch_norm(z)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        ff = ffR + ffT + ffS
        return ff

    elif activation == "onem2":
        # with tf.variable_scope(name) as scope:
        z = batch_norm(z)
        ffS = tf.sigmoid(z)
        ffT = tf.tanh(z)
        ffR = tf.nn.relu(z)
        ff = ffR + ffT + ffS
        return ff 

    elif activation == "onethreem1":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
        z = batch_norm(z)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        a1 = 1/3
        ff = a1*ffR + a1*ffT + a1*ffS
        return ff 

    elif activation == "onethreem2":
        z = batch_norm(z)
        ffS = tf.sigmoid(z)
        ffT = tf.tanh(z)
        ffR = tf.nn.relu(z)
        a1 = 1/3
        ff = a1*ffR + a1*ffT + a1*ffS
        return ff 


    #with tf.variable_scope(name) as scope:
    a = tf.Variable(tf.constant(1.0), name='a')
    b = tf.Variable(tf.constant(1.0), name='b')                 
    c = tf.Variable(tf.constant(0.0), name='c')
    d = tf.Variable(tf.constant(0.0), name='d')
    z = batch_norm(z)

    #a = tf.Variable(tf.constant(1.0), name='a')
    #b = tf.Variable(tf.constant(1.0), name='b')
    #c = tf.Variable(tf.constant(0.0), name='c')
    #d = tf.Variable(tf.constant(0.0), name='d')
    #alpha  =tf.Variable(tf.constant(0.0), name='alpha')
    #b = tf.Variable(tf.constant(1.0), name='b')

    #ff = tf.nn.leaky_relu(z, alpha=0.2, name=None)
    #pos = tf.nn.relu(z)
    #neg = alpha*(z- abs(z))*0.5
    #ff = b * tf.sigmoid(a * z + c) + d  ###AS
    #ff = b * tf.tanh(a * z + c) + d  ###AT
    #ff = tf.maximum(a*z+c,b*z+d) ###AR
   # ff = z*tf.sigmoid(b*z)
    # ff = tf.sigmoid(z)
   # ff = tf.nn.relu(z)
#    ff = tf.tanh(z)

    #return pos+neg
    # alpha = tf.Variable(tf.constant(1.0), name="alpha")
    # beta = tf.Variable(tf.constant(1.0), name="beta")
    # gamma = tf.Variable(tf.constant(1.0), name="gamma")
#    z = batch_norm(z, scope="bn" + name)
    #ff = tf.nn.leaky_relu(z, alpha=0.2, name=None)
    #alpha = tf.Variable(tf.constant(0.0), name='alpha')
    #pos = tf.nn.relu(z)
    #neg = alpha * (z - abs(z)) * 0.5
    ffT = b * tf.tanh(a * z + c) + d  ###AT
    ffR = tf.maximum(a*z+c,b*z+d) ###AR
    # ffR = tf.nn.relu(z)
   # ff = z * tf.sigmoid(b * z)
    ffS = b * tf.sigmoid(a * z + c) + d  ###AS
    # ffS = tf.sigmoid(z)

    # ffT = tf.tanh(z)
    #return pos        # fixed activation
    #return pos + neg
    # ff = alpha/(alpha + beta + gamma)*ffT + beta/(alpha + beta + gamma)*ffS + gamma/(alpha + beta + gamma)*ffR
    a1 = 1/3
    ff = a1*ffT + a1*ffS + a1*ffR
    
    return ff
    # return ff
    #return pos # fixed function.

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


class LeNet(object):
    def __init__(self, x, keep_prob, num_classes, activation = "relu"):
        self.x = x
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.activation = activation
        self._build_model()

    def _build_model(self):
        # Conv1 layer
        with tf.variable_scope('conv1') as scope:
            W_conv1 = weight_variable([5, 5, 3, 6])
            b_conv1 = bias_variable([6])
            h_conv1 = AdaptiveFunc(conv2d(self.x, W_conv1) + b_conv1,activation=self.activation)
        h_pool1 = max_pool_2x2(h_conv1, name='max_pool1')

        # Conv2 layer
        with tf.variable_scope('conv2') as scope:
            W_conv2 = weight_variable([14, 14, 6, 16])
            b_conv2 = bias_variable([16])
            h_conv2 = AdaptiveFunc(conv2d(h_pool1, W_conv2) + b_conv2 ,activation=self.activation)
        h_pool2 = max_pool_2x2(h_conv2, name='max_pool2')

        # FC layer
        with tf.variable_scope('fc_layer') as scope:
            W_fc1 = weight_variable([1024, 120])
            b_fc1 = bias_variable([120])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 1024])
            h_fc1 = AdaptiveFunc(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, activation=self.activation)

        # Drop layer
        with tf.variable_scope('drop_layer') as scope:
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Output layer
        with tf.variable_scope('output_layer') as scope:
            W_fc2 = weight_variable([120, self.num_classes])
            b_fc2 = bias_variable([self.num_classes])
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            print(y_conv.shape)
            self.output = y_conv
