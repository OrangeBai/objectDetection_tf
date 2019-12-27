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
        diff = y_true[:, :, :, 4 * num_anchors:] - y_pred
        valid = y_true[:, :, :, :4 * num_anchors]
        x_abs = bk.abs(diff)
        x_bool = bk.cast(bk.less_equal(x_abs, 1.0), tf.float32)

        loss = x_bool * (0.5 * diff * diff) + (1 - x_bool) * (x_abs - 0.5)
        return lambda_rpn_reg * bk.sum(valid * loss) / bk.sum(epsilon + valid)
    return rpn_loss_reg_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        anchor_valid = y_true[:, :, :, :num_anchors]
        anchor_signal = y_true[:, :, :, num_anchors:]
        valid_loss = bk.sum(anchor_valid * bk.binary_crossentropy(y_pred, anchor_signal))
        n_cls = bk.sum(epsilon + y_true[:, :, :, :num_anchors])
        return lambda_rpn_class * valid_loss / n_cls

    return rpn_loss_cls_fixed_num


def class_loss_reg(num_classes):
    def class_loss_reg_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        valid = y_true[:, :, : 4*num_classes]
        x_abs = bk.abs(x)
        x_bool = bk.cast(bk.less_equal(x_abs, 1.0), 'float32')
        loss = x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)
        return lambda_cls_reg * bk.sum(valid * loss) / bk.sum(epsilon + valid)
    return class_loss_reg_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * bk.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
