import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 

from tensorflow.python.training import moving_averages

def variable_weight(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)


def batch_norm(x, decay=0.999, epsilon=1e-03, scope="scope"):
    x_shape = x.get_shape()
    input_channels = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))

    with tf.variable_scope(scope):
        beta = variable_weight("beta", [input_channels, ],
                               initializer=tf.zeros_initializer())
        gamma = variable_weight("gamma", [input_channels, ],
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = variable_weight("moving_mean", [input_channels, ],
                                      initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = variable_weight("moving_variance", [input_channels],
                                          initializer=tf.ones_initializer(), trainable=False)

    mean, variance = tf.nn.moments(x, axes=reduce_dims)
    update_move_mean = moving_averages.assign_moving_average(moving_mean, mean, decay=decay)
    update_move_variance = moving_averages.assign_moving_average(moving_variance, variance, decay=decay)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


def AdaptiveFunc(z, name, activation):
    if activation == "relu":
        with tf.variable_scope(name) as scope:
            z = batch_norm(z, scope="bn" + name)
        ffR = tf.nn.relu(z)
        return ffR 

    elif activation == "tanh":
        with tf.variable_scope(name) as scope:
            z = batch_norm(z, scope="bn" + name)
        ffT = tf.tanh(z)
        return ffT

    elif activation == "sigmoid":
        with tf.variable_scope(name) as scope:
            z = batch_norm(z, scope="bn" + name)
        ffS = tf.sigmoid(z)
        return ffS

    elif activation == "arelu":
        with tf.variable_scope(name) as scope:
            a = tf.Variable(tf.constant(1.0), name='a')
            b = tf.Variable(tf.constant(1.0), name='b')
            c = tf.Variable(tf.constant(0.0), name='c')
            d = tf.Variable(tf.constant(0.0), name='d')
            z = batch_norm(z, scope="bn" + name)
        
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        return ffR

    elif activation == "atanh":
        with tf.variable_scope(name) as scope:
            a = tf.Variable(tf.constant(1.0), name='a')
            b = tf.Variable(tf.constant(1.0), name='b')
            c = tf.Variable(tf.constant(0.0), name='c')
            d = tf.Variable(tf.constant(0.0), name='d')
            z = batch_norm(z, scope="bn" + name)
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        return ffT

    elif activation == "asigmoid":
        with tf.variable_scope(name) as scope:
            a = tf.Variable(tf.constant(1.0), name='a')
            b = tf.Variable(tf.constant(1.0), name='b')
            c = tf.Variable(tf.constant(0.0), name='c')
            d = tf.Variable(tf.constant(0.0), name='d')
            z = batch_norm(z, scope="bn" + name)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        return ffS

    elif activation == "m1":
        with tf.variable_scope(name) as scope:
            a = tf.Variable(tf.constant(1.0), name='a')
            b = tf.Variable(tf.constant(1.0), name='b')
            c = tf.Variable(tf.constant(0.0), name='c')
            d = tf.Variable(tf.constant(0.0), name='d')
            alpha = tf.Variable(tf.constant(1.0), name="alpha")
            beta = tf.Variable(tf.constant(1.0), name="beta")
            gamma = tf.Variable(tf.constant(1.0), name="gamma")
            z = batch_norm(z, scope="bn" + name)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        ff = (alpha/(alpha + beta + gamma))*ffT + (beta/(alpha + beta + gamma))*ffS + (gamma/(alpha + beta + gamma))*ffR #m1
        return ff

    elif activation == "m2":
        with tf.variable_scope(name) as scope:
            alpha = tf.Variable(tf.constant(1.0), name="alpha")
            beta = tf.Variable(tf.constant(1.0), name="beta")
            gamma = tf.Variable(tf.constant(1.0), name="gamma")
            z = batch_norm(z, scope="bn" + name)
        ffS = tf.sigmoid(z)
        ffT = tf.tanh(z)
        ffR = tf.nn.relu(z)
        ff = (alpha/(alpha + beta + gamma))*ffT + (beta/(alpha + beta + gamma))*ffS + (gamma/(alpha + beta + gamma))*ffR #m1
        return  ff

    elif activation == "onem1":
        with tf.variable_scope(name) as scope:
            a = tf.Variable(tf.constant(1.0), name='a')
            b = tf.Variable(tf.constant(1.0), name='b')
            c = tf.Variable(tf.constant(0.0), name='c')
            d = tf.Variable(tf.constant(0.0), name='d')
            z = batch_norm(z, scope="bn" + name)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        ff = ffR + ffT + ffS
        return ff

    elif activation == "onem2":
        with tf.variable_scope(name) as scope:
            z = batch_norm(z, scope="bn" + name)
        ffS = tf.sigmoid(z)
        ffT = tf.tanh(z)
        ffR = tf.nn.relu(z)
        ff = ffR + ffT + ffS
        return ff 

    elif activation == "onethreem1":
        with tf.variable_scope(name) as scope:
            a = tf.Variable(tf.constant(1.0), name='a')
            b = tf.Variable(tf.constant(1.0), name='b')
            c = tf.Variable(tf.constant(0.0), name='c')
            d = tf.Variable(tf.constant(0.0), name='d')
            z = batch_norm(z, scope="bn" + name)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        a1 = 1/3
        ff = a1*ffR + a1*ffT + a1*ffS
        return ff 

    elif activation == "onethreem2":
        with tf.variable_scope(name) as scope:
            z = batch_norm(z, scope="bn" + name)
        ffS = tf.sigmoid(z)
        ffT = tf.tanh(z)
        ffR = tf.nn.relu(z)
        a1 = 1/3
        ff = a1*ffR + a1*ffT + a1*ffS
        return ff 

    
def conv_layer(x, filter_height, filter_width,
    num_filters, stride, name, padding = 'SAME', groups = 1, act = "relu"):
    input_channels = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape = [filter_height, filter_width, int(input_channels/groups), num_filters],
                initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01))
        # In the paper the biases of all of the layers have not been initialised the same way
        # name[4] gives the number of the layer whose weights are being initialised.
        if (name[4] == '1' or name[4] == '3'):
            b = tf.get_variable('biases', shape = [num_filters], 
                initializer = tf.constant_initializer(0.0))
        else:
            b = tf.get_variable('biases', shape = [num_filters], 
                initializer = tf.constant_initializer(1.0))

    if groups == 1:
        conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
    # In the cases of multiple groups, split inputs & weights
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis = 3, num_or_size_splits = groups, value = x)
        weight_groups = tf.split(axis = 3, num_or_size_splits = groups, value = W)
        output_groups = [tf.nn.conv2d(i, k, strides = [1, stride, stride, 1], padding = padding)
                        for i, k in zip(input_groups, weight_groups)]
        conv = tf.concat(axis = 3, values = output_groups)
    # Add the biases.
    z = tf.nn.bias_add(conv, b)
    # Apply ReLu non linearity.
    a = AdaptiveFunc(z,name, activation=act)
    return a

def fc_layer(x, input_size, output_size, name, relu = True, act="relu"):
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases.
        W = tf.get_variable('weights', shape = [input_size, output_size], 
            initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01))
        b = tf.get_variable('biases', shape = [output_size], 
            initializer = tf.constant_initializer(1.0))
        # Matrix multiply weights and inputs and add biases.
        z = tf.nn.bias_add(tf.matmul(x, W), b, name = scope.name)
    if relu:
        # Apply ReLu non linearity.
        a = AdaptiveFunc(z,name, activation=act)
        return a
    else:
        return z

def max_pool(x, name, filter_height = 3, filter_width = 3, stride = 2, padding = 'SAME'):
    return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
                        strides = [1, stride, stride, 1], padding = padding,
                        name = name)

def lrn(x, name, radius = 5, alpha = 1e-04, beta = 0.75, bias = 2.0):
    return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                                beta = beta, bias = bias, name = name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob = keep_prob)

class AlexNet(object):
    def __init__(self, x, keep_prob, num_classes, activation = "relu"):

        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.activation = activation
        self._build_model()


    def _build_model(self):
        
        # In the original implementation this would be:
        #conv1 = conv_layer(self.X, 11, 11, 96, 4, padding = 'VALID', name = 'conv1')
        print(self.X.shape)
        conv1 = conv_layer(self.X, 11, 11, 96, 2, name = 'conv1',act = self.activation)
        print(conv1.shape)
        norm1 = lrn(conv1, name = 'norm1')
        print(norm1.shape)
        pool1 = max_pool(norm1, padding = 'VALID', name = 'pool1')
        print(pool1.shape)

        conv2 = conv_layer(pool1, 5, 5, 256, 1, groups = 2, name = 'conv2',act = self.activation)
        print(conv2.shape)
        norm2 = lrn(conv2, name = 'norm2')
        print(norm2.shape)
        pool2 = max_pool(norm2, padding = 'VALID', name = 'pool2')
        print(pool2.shape)

        conv3 = conv_layer(pool2, 3, 3, 384, 1, name = 'conv3', act = self.activation)
        print(conv3.shape)

        # This conv. layer has been removed in this modified implementation
        # but is present in the original paper implementaion.
        conv4 = conv_layer(conv3, 3, 3, 384, 1, groups = 2, name = 'conv4', act = self.activation)
        print(conv4.shape)

        conv5 = conv_layer(conv4, 3, 3, 256, 1, groups = 2, name = 'conv5', act = self.activation)
        print(conv5.shape)
        pool5 = max_pool(conv5, padding = 'VALID', name = 'pool5')
        # print(conv1)
        print(pool5.shape)

        # In the original paper implementaion this will be:
        flattened = tf.reshape(pool5, [-1, 1 * 1 * 256])
        print(flattened.shape)
        fc6 = fc_layer(flattened, 1 * 1 * 256, 4096, name = 'fc6', act = self.activation) 
        print(fc6.shape)
        # flattened = tf.reshape(pool5, [-1, 1 * 1 * 256]) 
        # fc6 = fc_layer(flattened, 1 * 1 * 256, 1024, name = 'fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)
        print(dropout6.shape)

        # In the original paper implementaion this will be:
        fc7 = fc_layer(dropout6, 4096, 4096, name = 'fc7', act = self.activation)
        print(fc7.shape)
        # fc7 = fc_layer(dropout6, 1024, 2048, name = 'fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)
        print(dropout7.shape)

        # In the original paper implementaion this will be:
        fc8 = fc_layer(dropout7, 4096, self.NUM_CLASSES, relu = False, name = 'fc8')
        # fc8 = fc_layer(dropout7, 2048, self.NUM_CLASSES, relu = False, name = 'fc8')
        print(fc8.shape)
        self.output = fc8
