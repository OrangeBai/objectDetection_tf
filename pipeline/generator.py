import json
import random
import os
import cv2
import config as data_config
from configs.frcnn_config import *
import tensorflow as tf


class Generator(object):
    def __init__(self, net_config, mode='train'):
        self.img_shape = net_config.img_shape
        if mode == 'train':
            self.img_path = data_config.train_directory
            label_path = os.path.join(data_config.label_directory, 'train_label_new.json')
            with open(label_path, 'r') as f:
                self.label_list = json.load(f)
        elif mode == 'val':
            self.img_path = data_config.val_directory
            label_path = os.path.join(data_config.label_directory, 'val_label_new.json')
            with open(label_path, 'r') as f:
                self.label_list = json.load(f)
        else:
            self.img_path = data_config.test_directory
        self.cur_img = None

    def __next__(self):
        return self.next()

    def next(self):
        label_num = len(self.label_list)
        idx = random.randint(0, label_num - 1)
        cur_instance = self.label_list[idx]
        img_name = cur_instance['name']
        img_path = os.path.join(self.img_path, img_name)

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
