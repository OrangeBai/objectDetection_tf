from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import platform
import os

if platform == 'darwin':
    base_directory = r'/Users/oranbebai/Documents/Data/BDD Drive/bdd100k'
else:
    base_directory = r''

label_directory = os.path.join(base_directory, 'labels')


