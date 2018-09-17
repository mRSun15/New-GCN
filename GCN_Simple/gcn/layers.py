from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphLanzcosConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, node_num,placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphLanzcosConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # self.vars['weight_matrix'] = glorot([input_dim, output_dim],
            #                                             name='weights_matrix')
            for i in range(len(self.support)):
                # self.vars['weights_' + str(i)] = tf.Variable(tf.constant(i*3+1.0),name = 'weights_' + str(i))
                # print(self.vars['weights_' + str(i)])
                # tf.summary.scalar('weights_'+str(i), self.vars['weights_' + str(i)])
                self.vars['weights_'+str(i)] = glorot([input_dim, output_dim], name='weights_'+str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def Lanczos(self, operator, k, starting_vector = None):
        # with tf.name_scope(self.name):
        Vm = []
        beta = []
        alpha = []
        if self.sparse_inputs:
            v0 = tf.sparse_tensor_to_dense(starting_vector)
        else:
            v0 = starting_vector
        v0 = tf.nn.l2_normalize(tf.reshape(v0,[-1,1]))

        Vm.append(v0)
        for j in range(1, k):
            # print(j)
            w = dot(operator, Vm[j-1],sparse = True)
            alpha_j = dot(tf.transpose(w),Vm[j-1], sparse = False)
            new_v = w - tf.multiply(alpha_j, Vm[j-1])
            alpha.append(alpha_j)
            if j > 1:
                new_v -= tf.multiply(beta[j-2],Vm[j-2]) 
            
            beta_j = tf.norm(new_v)
            # print(new_v.shape)
            # print(beta_j.shape)
           
            beta.append(beta_j)
            new_v = tf.multiply(1.0/beta_j,new_v)
            Vm.append(new_v)
            # print("finish Lanczos")
        return Vm

    def _call(self, inputs):
        x = inputs
        # print(x.eval)
        dtype = x.dtype
        shape = tf.shape(x)
        # print(shape)
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        operators = self.support[0]
        supports = list()
        Vm_matrix = {}
        for j in range(len(self.support)):
            Vm_matrix[j] = []
        for i in range(self.input_dim):
            if self.sparse_inputs:
                cur_feature = tf.sparse_slice(x, [0,i],[self.node_num,1])
                # print(cur_feature.dense_shape)
                Vm = self.Lanczos(operators, len(self.support), cur_feature)
                for j in range(len(self.support)):
                    Vm_matrix[j].append(Vm[j])
            else:
                cur_feature = tf.slice(x, [0, i], [self.node_num, 1])
                Vm = self.Lanczos(operators, len(self.support), cur_feature)
                for j in range(len(self.support)):
                    Vm_matrix[j].append(Vm[j])
        print("finish Lanczos")
        for i in range(len(self.support)):
            print("concate: ", i) 
            pre_matrix = tf.concat(Vm_matrix[i], 1)
            support = dot(pre_matrix, self.vars['weights_'+str(i)], sparse=False)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        # print(output)
        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim,placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # self.vars['weight_matrix'] = glorot([input_dim, output_dim],
            #                                             name='weights_matrix')
            for i in range(len(self.support)):
                # self.vars['weights_' + str(i)] = tf.Variable(tf.constant(i*3+1.0),name = 'weights_' + str(i))
                # print(self.vars['weights_' + str(i)])
                # tf.summary.scalar('weights_'+str(i), self.vars['weights_' + str(i)])
                self.vars['weights_'+str(i)] = glorot([input_dim, output_dim], name='weights_'+str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_'+str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            # support = tf.scalar_mul(self.vars['weights_' + str(i)],support)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
