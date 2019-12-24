from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import platform
import os

if platform == 'darwin':
    base_directory = r'/Users/oranbebai/Documents/Data/BDD Drive/bdd100k'
else:
    base_directory = r'F:\DataSet\BDD100k\bdd100k'

label_directory = os.path.join(base_directory, 'labels')
train_directory = os.path.join(base_directory, 'images', '100k', 'train')
test_directory = os.path.join(base_directory, 'images', '100k', 'val')
val_directory = os.path.join(base_directory, 'images', '100k', 'val')

show_directory = os.path.join(base_directory, 'test')



