from model.frcnn import *
from pipeline.generator import *
from pipeline.drawer import *
from configs.frcnn_config import *
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.utils import Progbar
import config
from optparse import OptionParser

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
        print('Set memery growth False')


parser = OptionParser()

parser.add_option('-i', '--image_path_train', help='path to training data')
parser.add_option('-l', '--label_path_train', help='path to label')
parser.add_option('--val_image', help='valuation set image path')
parser.add_option('--val_label', help='Valuation set label path')
parser.add_option('--base_net', help='base net', default='res')
parser.add_option('--optimizer', help='optimizer', default='Adam')
parser.add_option('--lr', help='learning rate', default=1e-4)
parser.add_option('-w', '--weight_path', help='pre_trained weight path')
parser.add_option('-o', '--output_path', help='directory of output weights')
parser.add_option('-d', '--des', help='description of current work')
parser.add_option('--image_height', help='height of resized image', type=int)
parser.add_option('--image_width', help='width of resized image', type=int)

(options, args) = parser.parse_args()

des = options.des
if not options.output_path or not options.image_path_train or not options.label_path_train:
    print('Missing inputs')

img_shape = (options.image_height, options.image_width, 3)
gen = Generator(img_shape)
gen.bdd_parser(options.image_path_train, options.label_path_train)
c = FasterRcnnConfig(options.base_net, gen.num_cls, img_shape)
frcnn = FRCNN(c)

model_rpn, model_classifier, model_all = frcnn.train_model(options.weight_path, options.optimizer, options.lr)
model_rpn.summary()
num_epochs = 20
epoch_length = 200

losses = np.zeros((epoch_length, 6))

best_loss = np.Inf
for epoch_num in range(num_epochs):
    start_time = time.time()
    progbar = Progbar(epoch_length)
    for iter_num in range(epoch_length):
        try:
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

            elif num_neg < c.num_roi:
                neg_samples_size = num_neg
                selected_pos_samples = np.random.choice(pos_samples, c.num_roi - num_neg).tolist()
                selected_neg_samples = np.random.choice(neg_samples, num_neg).tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, c.num_roi // 2).tolist()
                selected_neg_samples = np.random.choice(neg_samples, c.num_roi // 2).tolist()

            selected_sample = []
            selected_sample.extend(selected_neg_samples)
            selected_sample.extend(selected_pos_samples)

            loss_class = model_classifier.train_on_batch([cur_img, roi_x[:, selected_sample, :]],
                                                         [roi_cls[:, selected_sample, :], roi_reg[:, selected_sample, :]])

            res = model_classifier.predict_on_batch([cur_img, roi_x[:, selected_sample, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]
            losses[iter_num, 5] = time.time() - start

            progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                          ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3]),
                                          ('rpn_positive', num_pos), ('time', losses[iter_num, 5])])
        except Exception as e:
            print('Exception: {}'.format(e))
            continue
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

    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
    file_name = '{0}_{1:02d}_loss-{2:.4f}.h5'.format(des, epoch_num, curr_loss)
    cur_output_path = os.path.join(options.output_path, file_name)
    print('Total loss: {0}, saving weights: {1}'.format(curr_loss, cur_output_path))

    model_all.save_weights(cur_output_path)
