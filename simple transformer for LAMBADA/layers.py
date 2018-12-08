import tensorflow as tf
from utils import mask_length
from attention import *


class Linear(object):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer=None, bias_initializer=None,
                 trainable=True, dtype=tf.float32, name=None):
        """
        Linear dense layer for sequential data.

        Args:
             units: the size of output representation
             activation: (Optional) layer activation
             use_bias: (Optional)
             kernel_initializer: (Optional)
             bias_initializer: (Optional)
             trainable: (Optional)
             dtype: (Optional)
             name: (Optional)

        """
        # general init
        if name is None:
            name = "Linear"
        self.name = str(name)
        self.variable_scope = tf.variable_scope(name)
        self._variables = []
        self._trainable_variables = {}
        self.built = False

        # importing inputs
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self._trainable = bool(trainable)
        self._dtype = dtype

        # setting attributes
        self.kernel = None
        self.bias = None
        self.inputs_dim = None

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, t):
        assert isinstance(t, bool)
        self._trainable = t
        for variable in self._trainable_variables:
            self._trainable_variables[variable] = t

    @property
    def variables(self):
        return self._variables

    @property
    def trainable_variables(self):
        return [var for var in self._variables if self._trainable_variables[var]]

    def build(self, input_shape):
        """
            Variables:
                kernel: a Tensor of shape [input_dim, self.units]
                bias: a Tensor of shape [self.units] if use_bias = True

        """
        self.inputs_dim = input_shape[-1]
        with tf.variable_scope(self.variable_scope):
            self.kernel = tf.get_variable(name="kernel", shape=[self.inputs_dim, self.units],
                                          dtype=self._dtype, initializer=self.kernel_initializer)
            self._variables.append(self.kernel)
            self._trainable_variables[self.kernel] = self.trainable
            if self.use_bias:
                self.bias = tf.get_variable(name="bias", shape=[self.units], dtype=self.dtype,
                                            initializer=self.bias_initializer)
                self._variables.append(self.bias)
                self._trainable_variables[self.bias] = self.trainable
        self.built = True

    def call(self, inputs, inputs_num=None, inputs_mask=None, name=None):
        """

        Args:
            inputs: a Tensor of shape [batch_size, seq_length, input_dim]
            inputs_num: (Optional) an integer Tensor of [batch_size] which specify length of
                        inputs for each sample
            inputs_mask: (Optional) a bool Tensor of shape [batch_size, seq_length] for specifying  the true
                         elements of inputs in the condition that inputs_num is not given
            name: (Optional)

        Returns:
            a Tensor of shape [batch_size, seq_length, self.units]

        """
        if name is None:
            name = "linear"

        with tf.name_scope(self.name):
            if inputs_num is not None and inputs_mask is not None:
                raise AttributeError("only one of inputs_num or inputs_mask should be specified")
            inputs_shape = tf.shape(inputs)
            batch_size = inputs_shape[0]
            seq_length = inputs_shape[1]
            inputs_dim = inputs_shape[2]
            tf.assert_equal(inputs_dim, self.inputs_dim)
            if inputs_num is not None:
                inputs_mask = mask_length(inputs_num, seq_length)
            if inputs_mask is None:
                inputs_mask = tf.fill([batch_size, seq_length], True)
            indices = tf.where(inputs_mask)
            inputs = tf.boolean_mask(inputs, inputs_mask)
            outputs = tf.matmul(inputs, self.kernel)
            if self.bias is not None:
                outputs = outputs + self.bias
            if self.activation is not None:
                outputs = self.activation(outputs)
            outputs = tf.scatter_nd(indices, outputs, [batch_size, seq_length, self.units])
        return tf.identity(outputs, name=name)

    def __call__(self, inputs, inputs_num=None, inputs_mask=None, name=None):
        """

        Args:
            inputs: a Tensor of shape [batch_size, seq_length, input_dim]
            inputs_num: (Optional) an integer Tensor of [batch_size] which specify length of
                        inputs for each sample
            inputs_mask: (Optional) a bool Tensor of shape [batch_size, seq_length] for specifying  the true
                         elements of inputs in the condition that inputs_num is not given
            name: (Optional)

        Returns:
            a Tensor of shape [batch_size, seq_length, self.units]

        """
        if not self.built:
            self.build(inputs.shape)
        return self.call(inputs, inputs_num, inputs_mask, name)


class SelfAttention(object):
    def __init__(self, units, initializer=None, trainable=True, name=None):
        """
        Self attention layer, give a sequence as input and apply bidirectional self-attention mechanism

        Args:
             units: the representation size of queries and keys
             initializer: (Optional) initializer for Query and Key map's weights
             trainable: (Optional)
             name: (Optional)
        """
        # general init
        if name is None:
            name = "self_attention"
        self.name = str(name)
        self.variable_scope = tf.variable_scope(self.name)
        self.layers = []
        self._variables = []
        self._trainable_variables = {}
        self.built = False

        # importing inputs
        self.units = units
        self.initializer = initializer
        self._trainable = bool(trainable)

        # setting attributes
        with self.variable_scope:
            self.key_layer = Linear(self.units, use_bias=False, kernel_initializer=self.initializer,
                                    trainable=self.trainable, name="key")
            self.query_layer = Linear(self.units, use_bias=False, kernel_initializer=self.initializer,
                                      trainable=self.trainable, name="query")
            self.layers.append(self.key_layer)
            self.layers.append(self.query_layer)
        self.inputs_dim = None

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, t):
        t = bool(t)
        self._trainable = t
        for variable in self._trainable_variables:
            self._trainable_variables[variable] = t
        for layer in self.layers:
            layer.trainable = t

    @property
    def variables(self):
        return self._variables + [layer.variables for layer in self.layers]

    @property
    def trainable_variables(self):
        return [var for var in self._variables if self._trainable_variables[var]] +\
               [layer.trainable_variables for layer in self.layers]

    def build(self, input_shape):
        self.inputs_dim = input_shape[-1]
        self.built = True

    def call(self, inputs, inputs_num=None, inputs_mask=None, name=None):
        """

        Args:
            inputs: a Tensor of shape [batch_size, seq_length, input_dim]
            inputs_num: (Optional) an integer Tensor of [batch_size] which specify length of
                        inputs for each sample
            inputs_mask: (Optional) a bool Tensor of shape [batch_size, seq_length] for specifying  the true
                         elements of inputs in the condition that inputs_num is not given
            name: (Optional)

        Returns:
            a Tensor of shape [batch_size, seq_length, self.units]

        """
        if name is None:
            name = "self_attention"

        with tf.name_scope(self.name):
            key = self.key_layer(inputs, inputs_num, inputs_mask)
            query = self.query_layer(inputs, inputs_num, inputs_mask)
            result = multiple_dot_attention(query, key, inputs, inputs_num, inputs_mask, inputs_num, inputs_mask)
        return tf.identity(result, name=name)

    def __call__(self, inputs, inputs_num=None, inputs_mask=None, name=None):
        """

        Args:
            inputs: a Tensor of shape [batch_size, seq_length, input_dim]
            inputs_num: (Optional) an integer Tensor of [batch_size] which specify length of
                        inputs for each sample
            inputs_mask: (Optional) a bool Tensor of shape [batch_size, seq_length] for specifying  the true
                         elements of inputs in the condition that inputs_num is not given
            name: (Optional)

        Returns:
            a Tensor of shape [batch_size, seq_length, self.units]

        """
        if not self.built:
            self.build(inputs.shape)
        return self.call(inputs, inputs_num, inputs_mask, name)













