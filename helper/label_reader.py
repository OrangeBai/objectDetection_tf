import os
import config
import json

train_labels = os.path.join(config.label_directory, 'bdd100k_labels_images_train.json')
with open(train_labels, 'r') as f:
    data_store = json.load(f)


def retrieve_data(num):
    cur_label = data_store[num]
    file_name = cur_label['name']
    attributes = cur_label['attributes']
    timestamp = cur_label['timestamp']
    labels = cur_label['labels']
    return file_name, attributes, timestamp, labels


