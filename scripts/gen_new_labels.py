import os
import config
import json
import random
from helper.label_reader import *
import cv2

train_labels = os.path.join(config.label_directory, 'bdd100k_labels_images_train.json')
val_labels = os.path.join(config.label_directory, 'bdd100k_labels_images_val.json')

new_labels_path_train = os.path.join(config.label_directory, 'train_label_new.json')
new_labels_path_val = os.path.join(config.label_directory, 'val_label_new.json')
cls_path = os.path.join(config.label_directory, 'cls.json')


with open(train_labels, 'r') as f:
    data_store_train = json.load(f)

with open(val_labels, 'r') as f:
    data_store_val = json.load(f)

new_label_train, cls_train = gen_new_label(data_store_train)
new_label_val, cls_val = gen_new_label(data_store_val)

assert list(cls_val.keys()).sort() == list(cls_val.keys()).sort()

with open(new_labels_path_train, 'w') as f:
    json.dump(new_label_train, f)

with open(new_labels_path_val, 'w') as f:
    json.dump(new_label_val, f)

with open(cls_path, 'w') as f:
    json.dump(cls_train, f)
