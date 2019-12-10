from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D
from tensorflow_core.python.keras.engine.input_layer import Input
import tensorflow as tf


def vgg_16_base(Input_tensor=None):
    # Block 1, 224*224*64
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2, 112*112*128
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3, 56*56*256
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4, 28*28*512
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5, 14*14*512
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # x: 7*7*512
    return x


input_tensor = Input(shape=(224, 224, 3))
x = vgg_16_base(input_tensor)
print(x.shape)
