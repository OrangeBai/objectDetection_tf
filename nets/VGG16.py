from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Flatten, Dense, Dropout
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from nets.RoiPoolingConv import *


def vgg_16_base(input_tensor=None):
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
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4, 28*28*512
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5, 14*14*512
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x: 7*7*512
    return x


def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_reg = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_reg, base_layers]


def classifier(base_layers, pooling_region, input_rois, num_rois, nb_classes):
    out_roi_pool = RoiPoolingConv(pooling_region, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_reg = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                              name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_reg]


if __name__ == '__main__':
    pass
