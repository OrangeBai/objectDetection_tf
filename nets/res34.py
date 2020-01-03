from __future__ import division
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, \
    BatchNormalization, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.keras.regularizers import l2
from nets.RoiPoolingConv import RoiPoolingConv

downscale = 16


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    num_filter1, num_filter2, num_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(num_filter1, (1, 1), padding='same', name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter3, (1, 1), padding='same', name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    num_filter1, num_filter2, num_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(num_filter1, (1, 1), trainable=trainable), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(num_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable),
                        name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(num_filter3, (1, 1), trainable=trainable), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, maxpooling=True, trainable=True):
    num_filter_1, num_filter_2, num_filter_3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(num_filter_1, (1, 1), padding='same',
               name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    if maxpooling:
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter_1, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filter_3, (1, 1), padding='same', name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(num_filter_3, (1, 1), name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
    if maxpooling:
        shortcut = MaxPooling2D((2, 2), strides=(2, 2))(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    num_filter_1, num_filter_2, num_filter_3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(num_filter_1, (1, 1), strides=strides, padding='same', trainable=trainable),
                        name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(num_filter_1, (kernel_size, kernel_size), padding='same',
                               trainable=trainable), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(num_filter_3, (1, 1), padding='same', trainable=trainable),
                        name=conv_name_base + '2c')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=3), name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)

    shortcut = TimeDistributed(Conv2D(num_filter_3, (1, 1), strides=strides, trainable=trainable),
                               name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def base(input_tensor, trainable=True):
    x = ZeroPadding2D((3, 3))(input_tensor)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', maxpooling=False, trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

    return x
