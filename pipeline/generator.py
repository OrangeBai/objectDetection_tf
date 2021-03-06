import json
import random
import os
import cv2
import config as data_config
from configs.frcnn_config import *
import tensorflow as tf
from bs4 import BeautifulSoup


class Generator(object):
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.img_data = []
        self.cur_img = None
        self.cls_dict = None
        self.num_cls = 0

    def __next__(self):
        return self.next()

    def bdd_parser(self, img_directory, label_file):
        cls = {'backgroud': 0}
        new_img_labels = []
        with open(label_file, 'r') as file:
            data_store = json.load(file)
        for img_label in data_store:
            new_label = {'path': os.path.join(img_directory, img_label['name']),
                         'attributes': img_label['attributes'],
                         'labels': []}
            for label in img_label['labels']:
                if 'box2d' in label.keys():
                    cur_label = {}
                    category = label['category']
                    if category not in cls.keys():
                        cls[category] = len(cls)
                    cur_label['category'] = cls[category]
                    cur_label['coordinates'] = (label['box2d']['x1'], label['box2d']['y1'],
                                                label['box2d']['x2'], label['box2d']['y2'])
                    cur_label['attributes'] = label['attributes']
                    new_label['labels'].append(cur_label)
            new_img_labels.append(new_label)
        self.img_data.extend(new_img_labels)
        self.cls_dict = cls
        self.num_cls = len(self.cls_dict.keys())

    def voc_parser(self, img_directory, label_directory):
        cls = {'backgroud': 0}
        label_files = os.listdir(label_directory)
        new_img_labels = []
        for label_file in label_files:
            new_img_label = {'path': '',
                             'labels': []}
            with open(os.path.join(label_directory, label_file)) as f:
                soup = BeautifulSoup(f, 'xml')
            new_img_label['path'] = os.path.join(img_directory, soup.filename.text)
            objs = soup.find_all('object')
            for obj in objs:
                cur_label = {}
                category = obj.find('name', recursive=False).text
                if category not in cls.keys():
                    cls[category] = len(cls)
                cur_label['category'] = cls[category]
                bbox = obj.find('bndbox', recursive=False)
                cur_label['coordinates'] = [int(bbox.xmin.text), int(bbox.ymin.text),
                                            int(bbox.xmax.text), int(bbox.ymax.text)]
                cur_label['truncated'] = int(obj.truncated.text)
                cur_label['difficult'] = int(obj.difficult.text)
                new_img_label['labels'].append(cur_label)
            new_img_labels.append(new_img_label)
            self.img_data.extend(new_img_labels)
            self.cls_dict = cls
            self.num_cls = len(cls.keys())
        return

    def next(self):
        label_num = len(self.img_data)
        idx = random.randint(0, label_num - 1)
        cur_instance = self.img_data[idx]
        img_path = cur_instance['path']

        self.cur_img = cv2.imread(img_path)
        labels = cur_instance['labels']
        if self.img_shape is not None:
            img, labels = self.resize_data(self.cur_img, labels)
        else:
            img = self.cur_img
        img = np.expand_dims(img, axis=0)
        img = tf.cast(img, tf.float32)
        return img, labels, img_path

    def resize_data(self, img, labels):
        new_img = cv2.resize(img, (self.img_shape[1], self.img_shape[0]))
        origin_shape = img.shape
        ratio_y = self.img_shape[0] / origin_shape[0]
        ratio_x = self.img_shape[1] / origin_shape[1]
        new_labels = []
        for label in labels:
            new_label = {'category': label['category'],
                         'coordinates': (label['coordinates'][0] * ratio_x,
                                         label['coordinates'][1] * ratio_y,
                                         label['coordinates'][2] * ratio_x,
                                         label['coordinates'][3] * ratio_y)}
            new_labels.append(new_label)

        return new_img, new_labels

    def retrieve_cls(self):
        return self.cls_dict, len(self.cls_dict.keys())

    def retrieve_cur_img(self):
        return self.cur_img


if __name__ == '__main__':
    pass
    # new_labels_path_train = os.path.join(config.label_directory, 'train_label_new.json')
    # gen = Generator(new_labels_path_train, (720, 1080))
    # for counter in range(20):
    #     cur_img, cur_label, path = next(gen)
    #     write_path = os.path.join(config.show_directory, str(counter) + '.jpg')
    #     for i in range(len(cur_label)):
    #         cv2.rectangle(cur_img, (int(cur_label[i]['coordinates'][0]), int(cur_label[i]['coordinates'][1])),
    #                       (int(cur_label[i]['coordinates'][2]), int(cur_label[i]['coordinates'][3])), (0, 255, 0))
    #     c = FasterRcnnConfig('vgg16')
    #     valid, signal, rpn_reg, cls, raw = c.cal_gt_tags(cur_img, cur_label, (22, 33))
    #     for ix in range(valid.shape[0]):
    #         for jy in range(valid.shape[1]):
    #             for idx in range(valid.shape[2]):
    #                 if signal[ix, jy, idx] == 1:
    #                     x1, y1, x2, y2 = raw[ix, jy, 4 * idx: 4*idx + 4]
    #                     cv2.rectangle(cur_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
    #     cv2.imwrite(write_path, cur_img)
    #     print(1)
