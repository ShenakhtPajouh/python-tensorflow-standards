import tensorflow as tf
from transformer import Transformer
from utils import mask_length


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding, num_blocks=3, transformer_units=100,trainable=True, name=None):
        """
        the model consist of an embedding layer, a transformer consist of num_blocks blocks and a softmax layer in the
        end which gives the probability vector of the blank word in the sentence. the model uses a trainable vector for
        blank embedding.

        Args:
            embedding: the word embedding for vocabs. a Tensor of shape [vocab_size, embedding_size]
            num_blocks: (Optional) an integer which specify the number of transformer blocks
            transformer_units: (Optional) an integer which shows the output representation.
            trainable: (Optional)
            name: (Optional)

        """
        assert num_blocks >= 0
        if name is None:
            name = "Linear"
        super().__init__(name=name)
        self._vocab_size = vocab_size
        self._embedding_size = embedding.shape[1]
        self._blank_embedding = None
        with tf.variable_scope(self.name):
            self._set_scope()
        with tf.variable_scope(self.name):
            with tf.name_scope(self.name):
                embedding_initializer = tf.keras.initializers.constant(embedding)
                self._embedding = tf.keras.layers.Embedding(vocab_size, self._embedding_size,
                                                            embedding_initializer, trainable=False)
                self._transformer = Transformer(transformer_units, num_blocks,
                                                trainable=self.trainable, name="transformer")
                self._softmax = tf.layers.Dense(vocab_size, tf.nn.softmax, trainable=self.trainable)
        self._layers = self._layers + [self._embedding] + [layer for layer in self._transformer.layers] +\
                       [self._softmax]

    def build(self, input_shape):
        self._blank_embedding = self.add_weight("blank_embedding", [self._embedding_size])

    def call(self, inputs, training=None, mask=None):
        """

        Args:
            inputs: a triple of tensors (inputs, blanks, num_inputs)
                    inputs: an integer Tensor of shape [batch_size, seq_length]
                    blanks: an integer Tensor of shape [batch_size] which specified the blank word in each sample
                    num_inputs: an integer Tensor of shape [batch_size] which specified the length of sequence in
                                each sample

        Returns:
            a Tensor of shape [batch_size, vocab_size] which is the probability vector for blank words

        """
        inputs, blanks, num_inputs = inputs
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        seq_length = inputs_shape[1]
        blanks_indices = tf.concat([tf.range(seq_length), blanks], 1)
        blank_mask = tf.scatter_nd(blanks_indices, tf.fill([batch_size]), [batch_size, seq_length])
        inputs_mask = mask_length(num_inputs)
        x = self._embedding(inputs)
        blanks_embedding = tf.zeros_like(x) + self._blank_embedding
        x = tf.where(blank_mask, blanks_embedding, x)
        x = self._transformer([x, inputs_mask])
        probs = self._softmax(x)
        return probs





