#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.training import moving_averages

def AdaptiveFunc(z, activation = "relu"):
    if activation == "relu":
        # with tf.variable_scope(name) as scope:
            # z = batch_norm(z, scope="bn" + name)
        # z = batch_norm(z)
        ffR = tf.nn.relu(z)
        return ffR 

    elif activation == "tanh":
        # with tf.variable_scope(name) as scope:
            # z = batch_norm(z, scope="bn" + name)
        # z = batch_norm(z)
        ffT = tf.tanh(z)
        return ffT

    elif activation == "sigmoid":
        # with tf.variable_scope(name) as scope:
        # z = batch_norm(z)
        ffS = tf.sigmoid(z)
        return ffS

    elif activation == "arelu":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
            # z = batch_norm(z, scope="bn" + name)
        # z = batch_norm(z)
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        return ffR

    elif activation == "atanh":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
            # z = batch_norm(z, scope="bn" + name)
        # z = batch_norm(z)
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        return ffT

    elif activation == "asigmoid":
        # with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')
        # z = batch_norm(z, scope="bn" + name)
        # z = batch_norm(z)
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
        # z = batch_norm(z)
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
        # z = batch_norm(z)
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
        # z = batch_norm(z)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        ff = ffR + ffT + ffS
        return ff

    elif activation == "onem2":
        # with tf.variable_scope(name) as scope:
        # z = batch_norm(z)
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
        # z = batch_norm(z)
        ffS = b * tf.sigmoid(a * z + c) + d  ###AS
        ffT = b * tf.tanh(a * z + c) + d  ###AT
        ffR = tf.maximum(a*z+c,b*z+d) ###AR
        a1 = 1/3
        ff = a1*ffR + a1*ffT + a1*ffS
        return ff 

    elif activation == "onethreem2":
        # z = batch_norm(z)
        ffS = tf.sigmoid(z)
        ffT = tf.tanh(z)
        ffR = tf.nn.relu(z)
        a1 = 1/3
        ff = a1*ffR + a1*ffT + a1*ffS
        return ff 


    


def AdaptiveFunc_A(z, name, activation = "relu"):
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

#     with tf.variable_scope(name) as scope:
#         a = tf.Variable(tf.constant(1.0), name='a')
#         b = tf.Variable(tf.constant(1.0), name='b')
#         c = tf.Variable(tf.constant(0.0), name='c')
#         d = tf.Variable(tf.constant(0.0), name='d')
#         alpha = tf.Variable(tf.constant(1.0), name="alpha")
#         beta = tf.Variable(tf.constant(1.0), name="beta")
#         gamma = tf.Variable(tf.constant(1.0), name="gamma")

#         z = batch_norm(z, scope="bn" + name)

#     #ff = tf.nn.leaky_relu(z, alpha=0.2, name=None)

#     #alpha = tf.Variable(tf.constant(0.0), name='alpha')
#     #pos = tf.nn.relu(z)
#     #neg = alpha * (z - abs(z)) * 0.5
#     #ff = z * tf.sigmoid(b * z)
#    # ff = b * tf.tanh(a * z + c) + d  ###AT
#     #ff = tf.maximum(a*z+c,b*z+d) ###AR
#     #ff = tf.nn.relu(z)
# #    ff = b * tf.sigmoid(a * z + c) + d  ###AS
#     #ff = tf.sigmoid(z)
#     #ff = tf.tanh(z)
#     #return pos + neg
#     #return ff


    
# #    z = batch_norm(z, scope="bn" + name)
#     #ff = tf.nn.leaky_relu(z, alpha=0.2, name=None)
#     #alpha = tf.Variable(tf.constant(0.0), name='alpha')
#     #pos = tf.nn.relu(z)
#     #neg = alpha * (z - abs(z)) * 0.5
#     ffT = b * tf.tanh(a * z + c) + d  ###AT
#     ffR = tf.maximum(a*z+c,b*z+d) ###AR
#     # ffR = tf.nn.relu(z)
#    # ff = z * tf.sigmoid(b * z)
#     ffS = b * tf.sigmoid(a * z + c) + d  ###AS
#     # ffS = tf.sigmoid(z)

#     # ffT = tf.tanh(z)
#     #return pos        # fixed activation
#     #return pos + neg
#     ff = alpha/(alpha + beta + gamma)*ffT + beta/(alpha + beta + gamma)*ffS + gamma/(alpha + beta + gamma)*ffR
#     return ff


def variable_weight(name, shape, initializer, trainable=True):

    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)

def conv_layer(x, num_outputs, kernel_size, stride=1, scope="conv2d"):

    input_channels = x.get_shape()[-1]

    with tf.variable_scope(scope):
        kernel = variable_weight("kernel", [kernel_size, kernel_size, input_channels, num_outputs], 
                tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")

def fc_layer(x, num_outputs, scope="fc"):

    input_channels = x.get_shape()[-1]
    
    with tf.variable_scope(scope):
        W = variable_weight("weight", [input_channels, num_outputs], 
            tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
        b = variable_weight("bias", [num_outputs,], 
            tf.zeros_initializer())
        return tf.nn.xw_plus_b(x, W, b)

# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, scope="scope"):

    x_shape = x.get_shape()
    input_channels = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))

    with tf.variable_scope(scope):
        beta = variable_weight("beta", [input_channels,], 
                                initializer=tf.zeros_initializer())
        gamma = variable_weight("gamma", [input_channels,], 
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = variable_weight("moving_mean", [input_channels,],
                                initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = variable_weight("moving_variance", [input_channels], 
                                initializer=tf.ones_initializer(), trainable=False)

    mean, variance = tf.nn.moments(x, axes=reduce_dims)
    update_move_mean = moving_averages.assign_moving_average(moving_mean, mean, decay=decay)
    update_move_variance = moving_averages.assign_moving_average(moving_variance, variance, decay=decay)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

def pool_layer(x, pool_size, pool_stride, name, padding='SAME', pooling_Mode='Max_Pool'):

    if pooling_Mode=='Max_Pool':
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1], [1, pool_stride, pool_stride, 1], padding = padding, name = name)

    if pooling_Mode=='Avg_Pool':
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1], [1, pool_stride, pool_stride, 1], padding = padding, name = name)

class ResNet50(object):

    def __init__(self, x, keep_prob, num_classes, activation = "relu"):

        self.X =x
        self.KEEP_PROB = keep_prob
        self.NUM_CLASSES = num_classes
        self.activation = activation
        self._build_model()

    def _build_model(self):

        conv1 = conv_layer(self.X, 64, 7, 2, scope="conv1")                                 # -> [batch, 112, 112, 64]
        bn1 = AdaptiveFunc(batch_norm(conv1, scope="bn1"), activation=self.activation)
        maxpool1 = pool_layer(bn1, 3, 2, name="maxpool1", pooling_Mode = 'Max_Pool')    # -> [batch, 56, 56, 64]

        block2 = self._block(maxpool1, 256, 3, init_stride=1, scope="block2")           # -> [batch, 56, 56, 256]

        block3 = self._block(block2, 512, 4, scope="block3")                            # -> [batch, 28, 28, 512]

        block4 = self._block(block3, 1024, 6, scope="block4")                           # -> [batch, 14, 14, 1024]

        block5 = self._block(block4, 2048, 3, scope="block5")                           # -> [batch, 7, 7, 2048]

        avgpool5 = pool_layer(block5, 7, 7, name="avgpool5", pooling_Mode = 'Avg_Pool')    # -> [batch, 1, 1, 2048]

        spatialsqueeze = tf.squeeze(avgpool5, [1, 2], name="SpatialSqueeze")            # -> [batch, 2048]

        self.logits = fc_layer(spatialsqueeze, self.NUM_CLASSES, "fc6")                 # -> [batch, num_classes]
        
        self.output = tf.nn.softmax(self.logits)

    def _block(self, x, n_out, n, init_stride=2, scope="block"):

        with tf.variable_scope(scope):
            h_out = n_out // 4
            out = self._bottleneck(x, h_out, n_out, stride=init_stride, scope="bottlencek1")
            for i in range(1, n):
                out = self._bottleneck(out, h_out, n_out, scope=("bottlencek%s" % (i + 1)))
            return out

    def _bottleneck(self, x, h_out, n_out, stride=None, scope="bottleneck"):

        input_channels = x.get_shape()[-1]

        if stride is None:
            stride = 1 if input_channels == n_out else 2

        with tf.variable_scope(scope):
            h = conv_layer(x, h_out, 1, stride=stride, scope="conv_1")
            h = batch_norm(h, scope="bn_1")
            h = AdaptiveFunc(h, activation=self.activation)
            h = conv_layer(h, h_out, 3, stride=1, scope="conv_2")
            h = batch_norm(h, scope="bn_2")
            h = AdaptiveFunc(h,activation=self.activation)
            h = conv_layer(h, n_out, 1, stride=1, scope="conv_3")
            h = batch_norm(h, scope="bn_3")

            if input_channels != n_out:
                shortcut = conv_layer(x, n_out, 1, stride=stride, scope="conv_4")
                shortcut = batch_norm(shortcut, scope="bn_4")
            else:
                shortcut = x

            return AdaptiveFunc(shortcut + h, activation=self.activation)
