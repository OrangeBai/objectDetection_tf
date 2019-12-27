import nets.VGG16 as vgg
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.training import Model


class ModelBase(object):
    def __init__(self):
        pass

    def build_model(self):
        pass


class FRCNN(ModelBase):
    def __init__(self, config, base_fun, rpn, cls_fun):
        super().__init__()
        self.num_class = config.num_cls
        self.num_anchor = config.num_anchors
        self.pooling_region = config.pooling_region
        self.num_roi = config.num_roi
        self.feature_shape = config.feature_shape

        self.img_input = Input(config.img_shape)
        self.roi_input = Input(shape=(None, 4))
        self.feature_input = Input(self.feature_shape)

        self.base_fun = base_fun
        self.rpn_fun = rpn
        self.cls_fun = cls_fun

    def train_model(self):
        base = self.base_fun(self.img_input)
        rpn = self.rpn_fun(base, self.num_anchor)
        classifier = self.cls_fun(base, self.pooling_region, self.roi_input, self.num_roi, nb_classes=self.num_class)

        model_rpn = Model(self.img_input, rpn[:2])
        model_classifier = Model([self.img_input, self.roi_input], classifier)
        model_all = Model([self.img_input, self.roi_input], rpn[:2] + classifier)

        return model_rpn, model_classifier, model_all

    def test_model(self):
        base = self.base_fun(self.img_input)
        rpn = self.rpn_fun(base, self.num_anchor)
        classifier = self.cls_fun(self.feature_input, self.pooling_region,
                                  self.roi_input, self.num_roi, nb_classes=self.num_class)

        model_rpn = Model(self.img_input, rpn)
        model_classifier_only = Model([self.feature_input, self.roi_input], classifier)
        return model_rpn, model_classifier_only


if __name__ == '__main__':
    pass

