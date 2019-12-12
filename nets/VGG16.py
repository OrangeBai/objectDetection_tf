from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.core import Flatten, Dense, Dropout
from tensorflow_core.python.keras.layers.wrappers import TimeDistributed
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.engine.training import Model
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
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x: 7*7*512
    return x


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    pooling_regions = 7
    input_shape = (num_rois, 7, 7, 512)

    out_roi_pool = RoiPoolingConv(7, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


if __name__ == '__main__':
    img_input = Input(shape=(224, 224, 3))
    roi_input = Input(shape=(None, 4))
    base = vgg_16_base(img_input)
    c = classifier(base, roi_input, 32, nb_classes=10, trainable=True)
    model_all = Model([img_input, roi_input], c)
    model_all.summary()

