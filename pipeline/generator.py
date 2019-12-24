import json
import random
import os
import cv2
import config
from configs.frcnn_config import *


class Generator(object):
    def __init__(self, file_path, reshape_size=None, mode='train'):
        self.file_path = file_path
        self.new_shape = reshape_size
        with open(file_path, 'r') as f:
            self.label_list = json.load(f)
        if mode == 'train':
            self.img_path = config.train_directory
        elif mode == 'val':
            self.img_path = config.val_directory
        else:
            self.img_path = config.test_directory

    def __next__(self):
        return self.next()

    def next(self):
        label_num = len(self.label_list)
        idx = random.randint(0, label_num - 1)
        cur_instance = self.label_list[idx]
        img_name = cur_instance['name']
        img_path = os.path.join(self.img_path, img_name)

        img = cv2.imread(img_path)
        labels = cur_instance['labels']
        if self.new_shape is not None:
            img, labels = self.resize_data(img, labels)
        return img, labels, img_path

    def resize_data(self, img, labels):
        new_img = cv2.resize(img, (self.new_shape[1], self.new_shape[0]))
        origin_shape = img.shape
        ratio_y = self.new_shape[0] / origin_shape[0]
        ratio_x = self.new_shape[1] / origin_shape[1]
        new_labels = []
        for label in labels:
            new_label = {'category': label['category'],
                         'coordinates': (label['coordinates'][0] * ratio_x,
                                         label['coordinates'][1] * ratio_y,
                                         label['coordinates'][2] * ratio_x,
                                         label['coordinates'][3] * ratio_y)}
            new_labels.append(new_label)
        return new_img, new_labels


if __name__ == '__main__':
    new_labels_path_train = os.path.join(config.label_directory, 'train_label_new.json')
    gen = Generator(new_labels_path_train, (720, 1080))
    for counter in range(20):
        cur_img, cur_label, path = next(gen)
        write_path = os.path.join(config.show_directory, str(counter) + '.jpg')
        for i in range(len(cur_label)):
            cv2.rectangle(cur_img, (int(cur_label[i]['coordinates'][0]), int(cur_label[i]['coordinates'][1])),
                          (int(cur_label[i]['coordinates'][2]), int(cur_label[i]['coordinates'][3])), (0, 255, 0))
        c = FasterRcnnConfig('vgg16')
        valid, signal, rpn_reg, cls, raw = c.cal_gt_tags(cur_img, cur_label, (22, 33))
        for ix in range(valid.shape[0]):
            for jy in range(valid.shape[1]):
                for idx in range(valid.shape[2]):
                    if signal[ix, jy, idx] == 1:
                        x1, y1, x2, y2 = raw[ix, jy, 4 * idx: 4*idx + 4]
                        cv2.rectangle(cur_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
        cv2.imwrite(write_path, cur_img)
        print(1)
