from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras import Model, Input

def ResUNet(input_layer):
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    return c5
  
import tensorflow.keras.backend as K

def weighted_binary_crossentropy(y_true, y_pred, weight_1=2.5):
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

  
input_layer = Input(shape=(2500, 2500, 13))  # Define the input layer
resunet_features = ResUNet(input_layer)

# Feature extractor output: 5 feature maps
feature_output = Conv2D(5, (1, 1), activation='linear', name='feature_output')(resunet_features)

# Ground truth classification output: 1 binary map
classification_output = Conv2D(1, (1, 1), activation='sigmoid', name='classification_output')(resunet_features)

# Define the model
model = Model(inputs=input_layer, outputs=[feature_output, classification_output])

model.compile(optimizer='adam',
              loss={'feature_output': None,  # No loss for feature output
                    'classification_output':
                        lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, weight_1=2.0)},
              loss_weights={'feature_output': 0.0,  # Do not focus on feature extraction
                            'classification_output': 1.0})


