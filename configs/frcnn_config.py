from configs.base_config import *
from helper.roi_helper import *
from nets import VGG16 as vgg16
import math
import numpy as np
from pipeline.generator import *


class FasterRcnnConfig(BaseConfig):
    def __init__(self, base_net):
        super().__init__()
        if base_net == 'vgg16':
            self.downscale = 32
            self.base_net = vgg16.vgg_16_base
            self.classifier = vgg16.classifier
        else:
            self.downscale = 32
        self.anchor_box_scale = [16, 32, 64, 128, 192, 256]
        self.anchor_box_ratio = [(1, 1), (1. / math.sqrt(2), 2. / math.sqrt(2)), (2. / math.sqrt(2), 1. / math.sqrt(2))]

        self.anchor_sets = [(scale, ratio) for scale in self.anchor_box_scale for ratio in self.anchor_box_ratio]
        self.rpn_positive = 0.6
        self.rpn_negative = 0.3
        self.num_cls = 10
        self.threshold = 0.7
        self.img_shape = (720, 1080)
        self.feature_shape = (22, 33)

    def cal_gt_tags(self, img, labels, feature_size):
        height = feature_size[0]
        width = feature_size[1]
        num_bbox = len(labels)
        nub_anchors = len(self.anchor_sets)

        box_valid = np.zeros((height, width, nub_anchors))
        box_signal = np.zeros((height, width, nub_anchors))
        box_class = np.zeros((height, width, nub_anchors))
        box_rpn_reg = np.zeros((height, width, 4 * nub_anchors))
        box_raw = np.zeros((height, width, 4 * nub_anchors))

        num_anchors_for_bbox = np.zeros(num_bbox).astype(int)

        best_anchor_for_bbox = -1 * np.ones((num_bbox, 3)).astype(int)
        best_iou_for_bbox = np.zeros(num_bbox).astype(np.float32)
        best_x_for_bbox = np.zeros((num_bbox, 4)).astype(int)
        best_dx_for_bbox = np.zeros((num_bbox, 4)).astype(np.float32)

        best_iou_all = 0
        for anchor_idx in range(len(self.anchor_sets)):

            anchor_scale, anchor_ratio = self.anchor_sets[anchor_idx]
            # the img in numpy_array form is [height, width], thus the x and y is opposite to that of coordinate system
            anchor_width = anchor_scale * anchor_ratio[0]
            anchor_height = anchor_scale * anchor_ratio[1]
            for ix in range(width):
                for jy in range(height):
                    x1 = (ix + 0.5) * self.downscale - anchor_width / 2
                    y1 = (jy + 0.5) * self.downscale - anchor_height / 2

                    x2 = x1 + anchor_width
                    y2 = y1 + anchor_height

                    if x1 < 0 or y1 < 0 or y2 > img.shape[0] or x2 > img.shape[1]:
                        continue

                    anchor_box = (x1, y1, x2, y2)
                    box_raw[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = [x1, y1, x2, y2]

                    best_iou = 0.0  # current best IoU for this anchor box
                    best_class = -1  # current best class for this anchor box
                    for idx in range(len(labels)):
                        label = labels[idx]
                        gt_class = label['category']
                        gt_box = label['coordinates']
                        cur_iou = calculate_iou(gt_box, anchor_box)
                        if cur_iou > best_iou_all:
                            best_iou_all = cur_iou
                        if cur_iou > self.rpn_positive and cur_iou > best_iou:
                            best_iou = cur_iou
                            best_class = gt_class

                            box_valid[jy, ix, anchor_idx] = 1
                            box_signal[jy, ix, anchor_idx] = 1
                            box_class[jy, ix, anchor_idx] = best_class

                            box_rpn_reg[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = cal_dx(gt_box, anchor_box)
                            num_anchors_for_bbox[idx] += 1

                        if cur_iou > best_iou_for_bbox[idx]:
                            best_iou_for_bbox[idx] = cur_iou
                            best_x_for_bbox[idx, :] = [x1, y1, x2, y2]

                            best_anchor_for_bbox[idx] = [jy, ix, anchor_idx]
                            best_dx_for_bbox[idx, :] = cal_dx(gt_box, anchor_box)

                        if cur_iou < self.rpn_negative:
                            box_valid[jy, ix, anchor_idx] = 1
                            box_signal[jy, ix, anchor_idx] = -1

        for bbox_idx in range(num_bbox):
            cur_jy, cur_ix, cur_anchor_idx = best_anchor_for_bbox[bbox_idx]
            box_valid[cur_jy, cur_ix, cur_anchor_idx] = 1
            box_signal[cur_jy, cur_ix, cur_anchor_idx] = 1
            box_rpn_reg[cur_jy, cur_ix, 4 * cur_anchor_idx: 4 * cur_anchor_idx + 4] = best_dx_for_bbox[bbox_idx]
            box_class[cur_jy, cur_ix, cur_anchor_idx] = labels[bbox_idx]['category']

        return box_valid, box_signal, box_rpn_reg, box_class, box_raw

    def select_highest_pred_box(self, pre_signal, pre_rpn_reg):
        rpn_width, rpn_height, num_anchors = pre_signal.shape[:3]
        num_all_anchors = rpn_width * rpn_height * num_anchors
        all_box = np.zeros((num_all_anchors, 5))
        all_prob = np.zeros(num_all_anchors)
        for ix in range(rpn_width):
            for jy in range(rpn_height):
                for anchor_idx in range(num_anchors):
                    cur_idx = ix * (rpn_height * num_anchors) + jy * num_anchors + anchor_idx
                    anchor_scale, anchor_ratio = self.anchor_sets[anchor_idx]

                    anchor_width = anchor_scale * anchor_ratio[0]
                    anchor_height = anchor_scale * anchor_ratio[1]

                    x1 = (ix + 0.5) * self.downscale - anchor_width / 2
                    y1 = (jy + 0.5) * self.downscale - anchor_height / 2

                    x2 = x1 + anchor_width
                    y2 = y1 + anchor_height

                    pred_box = inv_dx(pre_rpn_reg[ix, jy, 4 * anchor_idx: 4 * anchor_idx + 4], [x1, y1, x2, y2],
                                      self.img_shape)
                    all_box[cur_idx, :4] = pred_box
                    all_prob[cur_idx] = pre_signal[ix, jy, anchor_idx]

        selected = non_max_suppression_fast(all_box, all_prob)

        return selected

    def generate_classifier_data(self, selected, labels):
        anchor_boxes = selected[0]
        num_box = anchor_boxes.shape[0]

        anchor_cls = np.zeros((num_box, self.num_cls))
        anchor_reg = np.zeros((num_box, 4 * (self.num_cls - 1)))
        anchor_pos = np.zeros(num_box)

        anchor_x = np.zeros((num_box, 4))
        best_iou = 0
        for anchor_box_idx in range(num_box):
            cur_anchor_box = anchor_boxes[anchor_box_idx]
            feature_box = bbox_to_fbox(cur_anchor_box, self.downscale)
            anchor_x[anchor_box_idx, :] = clip_fbox(cal_x1_and_length(feature_box), self.feature_shape)

            for label in labels:
                cur_anchor_box = anchor_boxes[anchor_box_idx]
                cur_gt_box = label['coordinates']

                cur_gt_cls = label['category']

                iou = calculate_iou(cur_anchor_box, cur_gt_box)
                if iou > best_iou:
                    best_iou = iou

                if iou > self.threshold:
                    anchor_cls[anchor_box_idx, cur_gt_cls] = 1
                    normalized_anchor = fbox_to_bbox(feature_box, self.downscale)
                    anchor_reg[anchor_box_idx, 4 * cur_gt_cls - 4: 4 * cur_gt_cls] = cal_dx(cur_gt_box,
                                                                                            normalized_anchor)
                    anchor_pos[anchor_box_idx] = 1
                else:
                    anchor_cls[anchor_box_idx, 0] = 1
        anchor_x = np.expand_dims(anchor_x, axis=0)
        anchor_cls = np.expand_dims(anchor_cls, axis=0)
        anchor_reg = np.expand_dims(anchor_reg, axis=0)

        return anchor_x, anchor_cls, anchor_reg, anchor_pos
