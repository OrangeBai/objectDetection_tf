
class ModelBase(object):
    def __init__(self, img_size, nb_class):
        self.nb_class = nb_class
        self.img_size = img_size
        self.model = None

    def build_model(self):
        pass
