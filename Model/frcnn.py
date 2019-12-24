import nets.VGG16 as vgg
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.training import Model


class ModelBase(object):
    def __init__(self, image_shape, nb_class):
        self.nb_class = nb_class
        self.img_shape = image_shape
        self.model = None

    def build_model(self):
        pass


class FRCNN(ModelBase):
    def __init__(self, image_shape, nb_cls):
        super().__init__(image_shape, nb_cls)
        self.img_input = Input(self.img_shape)
        self.roi_input = Input(shape=(None, 4))
        self.pooling_region = 7
        self.model = None

    def create_model(self, base_fun, rpn, classifier):
        img_input = Input(self.img_shape)
        base = base_fun(img_input)
        rpn = rpn(base, 18)
        roi_input = Input(shape=(None, 4))
        c = classifier(base, self.pooling_region, roi_input, 32, nb_classes=10, trainable=True)
        model_rpn = Model(img_input, rpn[:2])
        model_classifier = Model([img_input, roi_input], c)
        model_all = Model([img_input, roi_input], rpn[:2] + c)

        return model_rpn, model_classifier, model_all


if __name__ == '__main__':
    shape = (720, 1080, 3)
    cls = 10
    frcnn = FRCNN(shape, cls)
    frcnn.create_model(vgg.vgg_16_base, vgg.rpn, vgg.classifier)

