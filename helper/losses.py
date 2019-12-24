import tensorflow.keras.backend as bk
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

lambda_rpn_reg = 1.0
lambda_rpn_class = 1.0

lambda_cls_reg = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_reg(num_anchors):
    def rpn_loss_reg_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, :] - y_pred
        x_abs = bk.abs(x)
        x_bool = bk.cast(bk.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_reg * bk.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / bk.sum(
            epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_reg_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * bk.sum(y_true[:, :, :, :] *
                                         bk.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, :])) \
               / bk.sum(epsilon + y_true[:, :, :, :])

    return rpn_loss_cls_fixed_num


def class_loss_reg(num_classes):
    def class_loss_reg_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = bk.abs(x)
        x_bool = bk.cast(bk.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_reg * bk.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / bk.sum(
            epsilon + y_true[:, :, :4 * num_classes])

    return class_loss_reg_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * bk.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
