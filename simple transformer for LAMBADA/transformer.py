import tensorflow as tf
from utils import mask_length
from layers2 import *


class Transformer(tf.keras.Sequential):
    def __init__(self, units, num_blocks, trainable=True, activation=tf.sigmoid, name=None):
        """
        simple transformer with self-attention and linear layers

        Args:
            units: an integer which shows the output representation
            num_blocks: an integer number of num_blocks

        """
        assert num_blocks >= 0
        if name is None:
            name = "Linear"
        super().__init__(name=name)
        with tf.variable_scope(self.name):
            self._set_scope()
        with tf.variable_scope(self.name):
            with tf.name_scope(self.name):
                for i in range(num_blocks):
                    self.add(SelfAttention(units, trainable=trainable, name="self_attention"))
                    self.add(Linear(units, activation=activation, trainable=trainable, name="dense"))
