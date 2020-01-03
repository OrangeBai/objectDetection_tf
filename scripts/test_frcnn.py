from model.frcnn import *
from pipeline.generator import *
from configs.frcnn_config import *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import helper.losses as losses
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.utils import Progbar

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
        print('Set memery growth False')

c = FasterRcnnConfig('vgg16')
c.num_cls=21
frcnn = FRCNN(c, vgg.vgg_16_base, vgg.rpn, vgg.classifier)
gen = Generator(c)

model_path = r"F:\Code\Computer Science\BDD-100k\model_path\00_loss-2.1877.h5"

model_rpn, model_classifier_only = frcnn.test_model()

model_rpn.load_weights(model_path, by_name=True)
model_classifier_only.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier_only.compile(optimizer='sgd', loss='mse')

for time in range(20):
    cur_img, cur_label, path = gen.next()
    rpn_cls, rpn_reg, cls, raw, valid = c.cal_gt_tags(cur_img, cur_label)
    cur_img = tf.cast(cur_img, tf.float32)

    signal_pre, rep_reg_pre, F = model_rpn.predict(cur_img)

    selected = c.select_highest_pred_box(signal_pre[0], rep_reg_pre[0])
    anchor_x, normalized_anchor = c.generate_test_data(selected)

    res = {'bounding':[], 'cls':[]}
    for i in range(anchor_x.shape[1] // c.num_roi):
        prob, dx = model_classifier_only.predict_on_batch([F, anchor_x[:, i * c.num_roi:i * c.num_roi + c.num_roi, :]])
        for j in range(c.num_roi):
            for idx in range(prob.shape[2]):
                if prob[0, j, idx] > 0.5:
                    if idx != 0:
                        pred_bounding = inv_dx(dx[0, j, idx * 4: idx * 4 + 4].numpy(), normalized_anchor[i * c.num_roi + j], c.img_shape)
                        pred_cls = idx
                        res['bounding'].append(pred_bounding)
                        res['cls'].append(pred_cls)
                    print(1)
    print(1)
