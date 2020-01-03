from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers import Conv2D, TimeDistributed, Dense,Flatten,Dropout
from nets.RoiPoolingConv import RoiPoolingConv
from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.optimizers import Adam, SGD
from helper import losses


class ModelBase(object):
    def __init__(self):
        pass

    def build_model(self):
        pass


class FRCNN(ModelBase):
    def __init__(self, config):
        super().__init__()
        self.num_class = config.num_cls
        self.num_anchor = config.num_anchors
        self.pooling_region = config.pooling_region
        self.num_rois = config.num_roi
        self.feature_shape = config.feature_shape

        self.img_input = Input(config.img_shape)
        self.roi_input = Input(shape=(None, 4))
        self.feature_input = Input(self.feature_shape)

        self.base_fun = config.base_net

        self.num_anchors = config.num_anchors
        self.num_cls = config.num_cls

    def rpn(self, base_layers):
        x = Conv2D(512, (3, 3), padding='same', activation='relu',
                   kernel_initializer='normal', name='rpn_conv1')(base_layers)

        x_class = Conv2D(self.num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', 
                         name='rpn_out_class')(x)
        x_regr = Conv2D(self.num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                        name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]

    def classifier(self, base_layers, trainable=True):
        out_roi_pool = RoiPoolingConv(self.pooling_region, self.num_rois)([base_layers, self.roi_input])

        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1', trainable=trainable))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2', trainable=trainable))(out)
        out = TimeDistributed(Dropout(0.5))(out)

        out_class = TimeDistributed(
            Dense(self.num_cls, activation='softmax', kernel_initializer='zero', trainable=trainable),
            name='dense_class_{}'.format(self.num_cls))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (self.num_cls - 1), activation='linear', kernel_initializer='zero',
                                         trainable=trainable), name='dense_regress_{}'.format(self.num_cls))(out)

        return [out_class, out_regr]

    def train_model(self, model_weight=None, optimizers=None, lr=None):

        base = self.base_fun(self.img_input)
        rpn = self.rpn(base)
        classifier = self.classifier(base)

        if lr is None:
            lr = 1e-4
        if optimizers == 'SGD':
            optimizer_rpn = SGD(lr=lr)
            optimizer_classifier = SGD(lr=lr)
        else:
            optimizer_rpn = Adam(lr=lr)
            optimizer_classifier = Adam(lr=lr)
        model_rpn = Model(self.img_input, rpn[:2])
        model_classifier = Model([self.img_input, self.roi_input], classifier)
        model_all = Model([self.img_input, self.roi_input], rpn[:2] + classifier)

        if model_weight is not None:
            model_rpn.load_weights(model_weight, by_name=True)
            model_classifier.load_weights(model_weight, by_name=True)
            model_all.load_weights(model_weight, by_name=True)

        model_rpn.compile(optimizer=optimizer_rpn,
                          loss=[losses.rpn_loss_cls(self.num_anchors), losses.rpn_loss_reg(self.num_anchors)])
        model_classifier.compile(optimizer=optimizer_classifier,
                                 loss=[losses.class_loss_cls, losses.class_loss_reg(self.num_cls - 1)],
                                 metrics={'dense_class_{}'.format(self.num_cls): 'accuracy'})
        model_all.compile(optimizer='sgd', loss='mae')

        return model_rpn, model_classifier, model_all

    def test_model(self):
        base = self.base_fun(self.img_input)
        rpn = self.rpn(base)
        classifier = self.classifier(base)

        model_rpn = Model(self.img_input, rpn)
        model_classifier_only = Model([self.feature_input, self.roi_input], classifier)
        return model_rpn, model_classifier_only


if __name__ == '__main__':
    pass
