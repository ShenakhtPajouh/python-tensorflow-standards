import tensorflow as tf
from attention import *

class Linear(tf.layers.Dense):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer=None, bias_initializer=None,
                 trainable=True, name=None):
        """
        Linear dense layer for sequential data.

        Args:
             units: the size of output representation
             activation: (Optional) layer activation
             use_bias: (Optional)
             kernel_initializer: (Optional)
             bias_initializer: (Optional)
             trainable: (Optional)
             name: (Optional)

        """
        # general init
        if name is None:
            name = "Linear"
        super().__init__(units, activation, use_bias, kernel_initializer, bias_initializer,
                         trainable=trainable, name=name)
        with tf.variable_scope(self.name):
            self._set_scope()
        self._inputs_dim = None

    def build(self, inputs_shape):
        self._inputs_dim = inputs_shape[0][-1]
        super().build([1, self._inputs_dim])

    def call(self, inputs):
        """

        Args:
            inputs: whether a Tensor of shape [batch_size, seq_length, input_dim] or a tuple of tensors, first a Tensor
            of shape [batch_size, seq_length, input_dim] which is inputs and a Tensor of type tf.bool and size of
            [batch_size, seq_length] for specifying the true elements of inputs


        Returns:
            a Tensor of shape [batch_size, seq_length, self.units]

        """
        if not isinstance(inputs, tf.Tensor):
            inputs_mask = None
        else:
            inputs, inputs_mask = inputs
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        seq_length = inputs_shape[1]
        inputs_dim = inputs_shape[2]
        indices = tf.where(inputs_mask)
        inputs = tf.boolean_mask(inputs, inputs_mask)
        outputs = super().call(inputs)
        outputs = tf.scatter_nd(indices, outputs, [batch_size, seq_length, self.units])
        return outputs


class SelfAttention(tf.keras.layers.Layer):
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
            name = "Linear"
        super().__init__(trainable, name)
        with tf.variable_scope(self.name):
            self._set_scope()
        self._units = units
        self._inputs_dim = None
        self._layers = []
        with tf.variable_scope(self.name):
            with tf.name_scope(self.name):
                self._key_dense = Linear(self._units, use_bias=False, kernel_initializer=initializer,
                                         trainable=self.trainable, name="key_dense")
                self._query_dense = Linear(self._units, use_bias=False, kernel_initializer=initializer,
                                           trainable=self.trainable, name="query_dense")
        self._layers = [self._key_dense, self._query_dense]

    @property
    def layers(self):
        return self._layers.copy()

    @property
    def variables(self):
        return sum([layer.variables for layer in self.layers], [])

    @property
    def trainable_variables(self):
        return sum([layer.trainable_variables for layer in self.layers], [])

    def call(self, inputs, **kwargs):
        """

        Args:
            inputs: whether a Tensor of shape [batch_size, seq_length, input_dim] or a tuple of tensors, first a Tensor
            of shape [batch_size, seq_length, input_dim] which is inputs and a Tensor of type tf.bool and size of
            [batch_size, seq_length] for specifying the true elements of inputs


        Returns:
            a Tensor of shape [batch_size, seq_length, self.units]

        """
        key = self._key_layer(inputs)
        query = self._query_layer(inputs)
        if not isinstance(inputs, tf.Tensor):
            inputs_mask = None
        else:
            inputs, inputs_mask = inputs
        result = multiple_dot_attention(query, key, inputs, query_mask=inputs_mask, memory_mask=inputs_mask)
        return result







