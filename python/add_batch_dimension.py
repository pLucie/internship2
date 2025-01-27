import tensorflow as tf

def add_batch_dimension(tensor):
    """
    Adds a batch dimension to the input tensor.

    Args:
        tensor (tf.Tensor): The input tensor to which the batch dimension will be added.

    Returns:
        tf.Tensor: The input tensor with an added batch dimension.
    """
    return tf.expand_dims(tensor, axis=0)
