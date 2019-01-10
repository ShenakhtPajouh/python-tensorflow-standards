import tensorflow as tf
from utils import mask_length


def simple_dot_attention(query, key, value, memory_length=None, memory_mask=None, name=None):
    """
    Attention method for given query, keys and values

    Args:
        query: a Tensor of shape [batch_size, query_dim]
        key: a Tensor of shape [batch_size, seq_length, query_dim]
        value: a Tensor of shape [batch_size, seq_length, value_dim]
        memory_length: (optional) an integer Tensor of shape [batch_size] which specify length of
                        memory (key and values) for each sample
        memory_mask: (optional) a bool Tensor of shape [batch_size, seq_length] for specifying the true elements of
                     memory in the condition that memory_length is not given
        name: (optional)

    Returns:
        a Tensor of shape [batch_size, value_dim] which is the result of attention mechanism

    """
    if name is None:
        name = "simple_dot_attention"
    with tf.name_scope(name):
        if memory_length is not None and memory_mask is not None:
            raise AttributeError("Only one of memory_length and memory_mask can be specified")
        query_shape = tf.shape(query)
        key_shape = tf.shape(key)
        value_shape = tf.shape(value)
        batch_size = query_shape[0]
        seq_length = key_shape[1]
        query_dim = query_shape[1]
        if memory_length is not None:
            memory_mask = mask_length(memory_length, seq_length)
        if memory_mask is None:
            memory_mask = tf.fill([batch_size, seq_length], True)
        indices = tf.where(memory_mask)
        queries = tf.gather(query, indices[:, 0])
        keys = tf.boolean_mask(key, memory_mask)
        attention_logits = tf.reduce_sum(queries * keys)
        attention_logits = tf.scatter_nd(tf.where(memory_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(memory_mask, attention_logits, tf.fill([batch_size, seq_length], -float("Inf")))
        attention_coefficients = tf.nn.softmax(attention_logits)
        attention = tf.expand_dims(attention_coefficients, -1) * value
    return tf.reduce_sum(attention, 1, name=name)


def multiple_dot_attention(query, key, value, query_length=None, query_mask=None, memory_length=None,
                           memory_mask=None, name=None):
    """
    Attention method for given queries, keys and values, which in each sample we have multiple queries.
    (a sequence of queries)

    Args:
        query: a Tensor of shape [batch_size, q_length, query_dim]
        key: a Tensor of shape [batch_size, seq_length, query_dim]
        value: a Tensor of shape [batch_size, seq_length, value_dim]
        query_length: (optional) an integer Tensor of shape [batch_size] which specify length of
                        queries for each sample
        query_mask: (optional) a bool Tensor of shape [batch_size, query_length] for specifying  the true
                    elements of queries in the condition that query_length is not given
        memory_length: (optional) an integer Tensor of shape [batch_size] which specify length of
                        memory (key and values) for each sample
        memory_mask: (optional) a bool Tensor of shape [batch_size, seq_length] for specifying the true elements of
                     keys and values in the condition that memory_length is not given
        name: (optional)

    Returns:
        a Tensor of shape [batch_size, q_length, value_dim] which is the result of attention mechanism

    """
    if name is None:
        name = "multiple_dot_attention"
    with tf.name_scope(name):
        if query_length is not None and query_mask is not None:
            raise AttributeError("Only one of query_length and query_mask can be specified")
        if memory_length is not None and memory_mask is not None:
            raise AttributeError("Only one of memory_length and memory_mask can be specified")
        query_shape = tf.shape(query)
        key_shape = tf.shape(key)
        value_shape = tf.shape(value)
        batch_size = query_shape[0]
        q_length = query_shape[1]
        seq_length = key_shape[1]
        query_dim = query_shape[2]
        value_dim = value_shape[2]
        if query_length is not None:
            query_mask = mask_length(query_length, q_length)
        if query_mask is None:
            query_mask = tf.fill([batch_size, q_length], True)
        if memory_length is not None:
            memory_mask = mask_length(memory_length, seq_length)
        if memory_mask is None:
            memory_mask = tf.fill([batch_size, seq_length], True)
        indices = tf.where(query_mask)
        query = tf.boolean_mask(query, query_mask)
        key = tf.gather(key, indices[:, 0])
        value = tf.gather(value, indices[:, 0])
        memory_mask = tf.gather(memory_mask, indices[:, 0])
        attention = simple_dot_attention(query, key, value, memory_mask=memory_mask)
    return tf.scatter_nd(indices, attention, [batch_size, q_length, value_dim], name=name)























