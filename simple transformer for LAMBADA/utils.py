import tensorflow as tf

def mask_length(length_tensor, max_length, name=None):
    """
    A method which gives the boolean mask tensor related to length_tensor. for length_tensor=[2, 1, 0, 1] and
    max_length=3 it returns [[True, True, False], [True, False, False], [False, False, False], [True, False, False]]

    Args:
        length_tensor: a non-negative integer Tensor with elements less than last dim size
        max_length:a Scalar

    Returns:
         a Tensor of shape [...length_tensor shape..., max_length] of type tf.bool

    Raises:


    """
    if name is None:
        name = "max_length"
    with tf.name_scope(name):
        length_tensor = tf.expand_dims(length_tensor, -1)
        ranges = tf.zeros_like(length_tensor, dtype=length_tensor.dtype) + tf.range(max_length, dtype=length_tensor.dtype)
    return tf.less(ranges, length_tensor, name=name)





