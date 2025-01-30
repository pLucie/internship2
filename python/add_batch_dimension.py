import tensorflow as tf

#this function will add a batch dimension to a tensor. 
def add_batch_dimension(tensor):

    return tf.expand_dims(tensor, axis=0)
