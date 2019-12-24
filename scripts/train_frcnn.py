from Model.frcnn import *
from pipeline.generator import *
from configs.frcnn_config import *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import helper.losses as losses
import numpy as np
import tensorflow as tf

new_labels_path_train = os.path.join(config.label_directory, 'train_label_new.json')
gen = Generator(new_labels_path_train, (720, 1080))
c = FasterRcnnConfig('vgg16')

shape = (720, 1080, 3)
cls = 10
frcnn = FRCNN(shape, cls)
model_rpn, model_classifier, model_all = frcnn.create_model(vgg.vgg_16_base, vgg.rpn, vgg.classifier)

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(18), losses.rpn_loss_reg(18)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_reg(10)],
                         metrics={'dense_class_{}'.format(10): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')
model_rpn.summary(120)

for counter in range(20):
    cur_img, cur_label, path = next(gen)
    valid, signal, rpn_reg, cls, raw = c.cal_gt_tags(cur_img, cur_label, (22, 33))
    signal = np.expand_dims(signal, axis=0)
    rpn_reg = np.expand_dims(rpn_reg, axis=0)
    cur_img = np.expand_dims(cur_img, axis=0)
    cur_img = tf.cast(cur_img, tf.float32)

    model_rpn.train_on_batch(cur_img, [signal, rpn_reg])

    signal_pre, rep_reg_pre = model_rpn.predict_on_batch(cur_img)

    c.select_highest_pred_box(signal_pre[0].numpy(), rep_reg_pre[0].numpy())
    print(1)
