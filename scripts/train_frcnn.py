from Model.frcnn import *
from pipeline.generator import *
from pipeline.drawer import *
from configs.frcnn_config import *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import helper.losses as losses
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.utils import Progbar
import config

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
        print('Set memery growth False')

c = FasterRcnnConfig('vgg16')
frcnn = FRCNN(c, vgg.vgg_16_base, vgg.rpn, vgg.classifier)
gen = Generator(c)

model_path = r'F:\PHD\Computer science\RA\BDD-100k\vgg16_raw_0.hdf5'

model_rpn, model_classifier, model_all = frcnn.train_model()

# model_rpn.load_weights(model_path, by_name=True)
# model_classifier.load_weights(model_path, by_name=True)
# model_all.load_weights(model_path, by_name=True)

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-6)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(c.num_anchors), losses.rpn_loss_reg(c.num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_reg(c.num_cls - 1)],
                         metrics={'dense_class_{}'.format(10): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')
model_rpn.summary(120)

num_epochs = 20
epoch_length = 200

losses = np.zeros((epoch_length, 6))

best_loss = np.Inf
for epoch_num in range(num_epochs):
    start_time = time.time()
    progbar = Progbar(epoch_length)
    for iter_num in range(epoch_length):
        start = time.time()

        cur_img, cur_label, path = gen.next()

        rpn_cls, rpn_reg, cls, raw, valid = c.cal_gt_tags(cur_img, cur_label)

        if iter_num < 10:
            d = Drawer(gen.retrieve_cur_img(), c.img_shape, os.path.join(config.show_directory, str(iter_num) + '.jpg'))
            d.draw_labels(cur_label)
            d.draw_gt_labels(rpn_cls, rpn_reg, raw)
            d.paint()

        loss_rpn = model_rpn.train_on_batch(cur_img, [rpn_cls, rpn_reg])
        signal_pre, rep_reg_pre = model_rpn.predict_on_batch(cur_img)

        selected = c.select_highest_pred_box(signal_pre[0].numpy(), rep_reg_pre[0].numpy())
        roi_x, roi_cls, roi_reg, roi_pos = c.generate_classifier_data(selected, cur_label)

        pos_samples = np.where(roi_pos == 1)[0]
        neg_samples = np.where(roi_pos == 0)[0]

        num_pos = len(pos_samples)
        num_neg = len(neg_samples)

        if num_pos < c.num_roi:
            pos_samples_size = num_pos
            selected_pos_samples = np.random.choice(pos_samples, num_pos).tolist()
            selected_neg_samples = np.random.choice(neg_samples, c.num_roi - num_pos).tolist()

        # elif num_neg < c.num_roi:
        #     neg_samples_size = num_neg
        #     selected_pos_samples = np.random.choice(pos_samples, c.num_roi - num_neg).tolist()
        #     selected_neg_samples = np.random.choice(neg_samples, num_neg).tolist()
        else:
            selected_pos_samples = np.random.choice(pos_samples, c.num_roi - 2).tolist()
            selected_neg_samples = np.random.choice(neg_samples, 2).tolist()

        selected_sample = []
        selected_sample.extend(selected_neg_samples)
        selected_sample.extend(selected_pos_samples)

        loss_class = model_classifier.train_on_batch([cur_img, roi_x[:, selected_sample, :]],
                                                     [roi_cls[:, selected_sample, :], roi_reg[:, selected_sample, :]])

        losses[iter_num, 0] = loss_rpn[1]
        losses[iter_num, 1] = loss_rpn[2]

        losses[iter_num, 2] = loss_class[1]
        losses[iter_num, 3] = loss_class[2]
        losses[iter_num, 4] = loss_class[3]
        losses[iter_num, 5] = time.time() - start

        progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                      ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3]),
                                      ('rpn_positive', num_pos), ('current_time', losses[iter_num, 5])])

    loss_rpn_cls = np.mean(losses[:, 0])
    loss_rpn_regr = np.mean(losses[:, 1])
    loss_class_cls = np.mean(losses[:, 2])
    loss_class_regr = np.mean(losses[:, 3])
    class_acc = np.mean(losses[:, 4])

    rpn_accuracy_for_epoch = []

    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
    print('Loss RPN regression: {}'.format(loss_rpn_regr))
    print('Loss Detector classifier: {}'.format(loss_class_cls))
    print('Loss Detector regression: {}'.format(loss_class_regr))
    print('Elapsed time: {}'.format(time.time() - start_time))
    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

    if curr_loss < best_loss:
        best_loss = curr_loss
        model_all.save_weights(model_path)
