import tensorflow.keras.backend as K

def weighted_binary_crossentropy(y_true, y_pred, weight_1=2.0):
    """
    Weighted binary crossentropy loss function.

    Parameters:
    - y_true: Ground truth values.
    - y_pred: Predicted values.
    - weight_1: Weight for the positive class (1).

    Returns:
    - Weighted binary crossentropy loss.
    """
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    # Compute binary crossentropy
    bce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    # Apply weights
    weighted_bce = bce * (y_true * weight_1 + (1 - y_true))
    return K.mean(weighted_bce)
