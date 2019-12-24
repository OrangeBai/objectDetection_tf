from Model.frcnn import *
from pipeline.generator import *
from configs.frcnn_config import *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import helper.losses as losses
import numpy as np
import tensorflow as tf
import time

roi_num = 32
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

for counter in range(100):
    t0 = time.time()
    cur_img, cur_label, path = next(gen)
    valid, signal, rpn_reg, cls, raw = c.cal_gt_tags(cur_img, cur_label, (22, 33))
    signal = np.expand_dims(signal, axis=0)
    rpn_reg = np.expand_dims(rpn_reg, axis=0)
    cur_img = np.expand_dims(cur_img, axis=0)
    cur_img = tf.cast(cur_img, tf.float32)

    t1 = time.time()
    model_rpn.train_on_batch(cur_img, [signal, rpn_reg])
    print('RPN train_time: {0}\n'.format(t1-t0))
    signal_pre, rep_reg_pre = model_rpn.predict_on_batch(cur_img)
    t2 = time.time()
    print('RPN predict_time: {0}\n'.format(t2 - t1))

    selected = c.select_highest_pred_box(signal_pre[0].numpy(), rep_reg_pre[0].numpy())
    roi_x, roi_cls, roi_reg, roi_pos = c.generate_classifier_data(selected, cur_label)

    pos_samples = np.where(roi_pos == 1)[0]
    neg_samples = np.where(roi_pos == 0)[0]

    if len(pos_samples) < roi_num // 2:
        pos_samples_size = len(pos_samples)
        selected_pos_samples = np.random.choice(pos_samples, pos_samples_size).tolist()
        selected_neg_samples = np.random.choice(neg_samples, roi_num - pos_samples_size).tolist()

    elif len(neg_samples) < roi_num // 2:
        neg_samples_size = len(neg_samples)
        selected_pos_samples = np.random.choice(pos_samples, roi_num - neg_samples_size).tolist()
        selected_neg_samples = np.random.choice(neg_samples, neg_samples).tolist()
    else:
        selected_pos_samples = np.random.choice(pos_samples, roi_num // 2).tolist()
        selected_neg_samples = np.random.choice(neg_samples, roi_num // 2).tolist()

    selected_sample = []
    selected_sample.extend(selected_neg_samples)
    selected_sample.extend(selected_pos_samples)

    t3 = time.time()
    print('Generate label time: {0}\n'.format(t3 - t2))
    model_classifier.train_on_batch([cur_img, roi_x[:, selected_sample, :]],
                                    [roi_cls[:, selected_sample, :], roi_reg[:, selected_sample, :]])
    t4 = time.time()
    print('Classifier train time: {0}\n'.format(t4 - t3))
    print(1)
