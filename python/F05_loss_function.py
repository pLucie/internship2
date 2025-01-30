import tensorflow.keras.backend as K
import tensorflow as tf

def weighted_f05_loss(y_true, y_pred, weight_1=2.0):
    #obtain a small value
    epsilon = tf.keras.backend.epsilon()

    # Clip predictions to avoid mathematical errors by dividing by zero.
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Calculate precision and recall
    precision = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_pred) + epsilon)
    recall = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + epsilon)

    # Calculate F0.5 score
    f05 = (1 + 0.5 ** 2) * (precision * recall) / (0.5 ** 2 * precision + recall + epsilon)

    # Weighted loss
    loss = -tf.reduce_mean(f05)  # Negative F0.5 score to minimize it during training
    return loss
