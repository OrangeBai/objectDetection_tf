from configs.base_config import *
from helper.roi_helper import *
from nets import VGG16 as vgg16
import math
import numpy as np
from pipeline.generator import *


class FasterRcnnConfig(BaseConfig):
    def __init__(self, base_net):
        """

        @param base_net: Name of base net
        @param img_shape: Shape of input image (height, width)
        @param feature_shape: Feature Map shape, adjust according to base net (height, width)
        @param num_cls: classification class
        """
        super().__init__()
        if base_net == 'vgg16':
            self.downscale = 32
            self.base_net = vgg16.vgg_16_base
            self.classifier = vgg16.classifier
        else:
            self.downscale = 16
        self.anchor_box_scale = [8, 16, 32, 64, 128, 256]
        self.anchor_box_ratio = [(1, 1), (1. / math.sqrt(2), 2. / math.sqrt(2)), (2. / math.sqrt(2), 1. / math.sqrt(2)),
                                 (1, 2), (2, 1)]
        self.anchor_sets = [(scale, ratio) for scale in self.anchor_box_scale for ratio in self.anchor_box_ratio]

        self.num_anchors = len(self.anchor_sets)
        self.rpn_positive = 0.45
        self.rpn_negative = 0.2
        self.num_cls = 10
        self.threshold = 0.5
        self.img_shape = (720, 960, 3)
        self.feature_shape = (45, 60, 512)
        self.pooling_region = 5
        self.num_roi = 8

    def cal_gt_tags(self, img, labels):
        """
        Calculate image ground truth value according to net config
        @param img: the input img
        @param labels: labels of the image
        @return: ground truth tag for rpn net
        """
        f_height, f_width = self.feature_shape[:2]  # feature map size
        num_bbox = len(labels)  # number of objects
        num_anchors = len(self.anchor_sets)

        box_valid = np.zeros((f_height, f_width, num_anchors))  # whether anchor box is valid(positive or negative)
        box_signal = np.zeros((f_height, f_width, num_anchors))  # anchor box label(0 for negative, 1 for positive)
        box_class = np.zeros((f_height, f_width, num_anchors))  # class for the box

        box_rpn_valid = np.zeros((f_height, f_width, 4 * num_anchors))  # whether rpn regression valid
        box_rpn_reg = np.zeros((f_height, f_width, 4 * num_anchors))  # rpn regression ground truth

        box_raw = np.zeros((f_height, f_width, 4 * num_anchors))  # raw anchor box without adjust

        num_anchors_for_bbox = np.zeros(num_bbox).astype(int)  # positive anchor for each bounding box

        # find the best anchor box for each bounding box, make sure there are at least 1 positive anchor box
        best_anchor_for_bbox = -1 * np.ones((num_bbox, 3)).astype(int)
        best_iou_for_bbox = np.zeros(num_bbox).astype(np.float32)
        best_x_for_bbox = np.zeros((num_bbox, 4)).astype(int)
        best_dx_for_bbox = np.zeros((num_bbox, 4)).astype(np.float32)

        # best iou of all boxes
        best_iou_all = 0
        for anchor_idx in range(len(self.anchor_sets)):

            anchor_scale, anchor_ratio = self.anchor_sets[anchor_idx]

            anchor_width = anchor_scale * anchor_ratio[0]
            anchor_height = anchor_scale * anchor_ratio[1]
            for ix in range(f_width):
                for jy in range(f_height):
                    # calculate anchor box coordinates
                    x1 = (ix + 0.5) * self.downscale - anchor_width / 2
                    y1 = (jy + 0.5) * self.downscale - anchor_height / 2

                    x2 = x1 + anchor_width
                    y2 = y1 + anchor_height
                    # If anchor box invalid, continue
                    x1, y1, x2, y2 = clip_box([x1, y1, x2, y2], self.img_shape[:2])

                    # anchor box coordinates in raw image
                    anchor_box = (x1, y1, x2, y2)
                    box_raw[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = [x1, y1, x2, y2]

                    best_iou = 0.0  # current best IoU for this anchor box
                    best_class = -1  # current best class for this anchor box
                    for label_idx in range(len(labels)):
                        # for each object, calculate the iou between current anchor box and object
                        label = labels[label_idx]
                        gt_class = label['category']
                        gt_box = label['coordinates']
                        cur_iou = calculate_iou(gt_box, anchor_box)
                        if cur_iou > best_iou_all:
                            # record best iou of all anchors for debugging
                            best_iou_all = cur_iou
                        if cur_iou > self.rpn_positive and cur_iou > best_iou:
                            # if higher than its best, box valid, label it positive
                            best_iou = cur_iou
                            best_class = gt_class

                            box_valid[jy, ix, anchor_idx] = 1
                            box_signal[jy, ix, anchor_idx] = 1
                            box_class[jy, ix, anchor_idx] = best_class

                            # calculate the rpn regression ground truth
                            box_rpn_valid[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = 1
                            box_rpn_reg[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = cal_dx(gt_box, anchor_box)
                            num_anchors_for_bbox[label_idx] += 1

                        if cur_iou > best_iou_for_bbox[label_idx]:
                            # id1f higher than the best iou of current object, record
                            best_iou_for_bbox[label_idx] = cur_iou
                            best_x_for_bbox[label_idx, :] = [x1, y1, x2, y2]

                            best_anchor_for_bbox[label_idx] = [jy, ix, anchor_idx]
                            best_dx_for_bbox[label_idx, :] = cal_dx(gt_box, anchor_box)

                    if best_iou < self.rpn_negative:
                        # if best iou of current anchor box less than threshold, box valid, label it negative
                        box_valid[jy, ix, anchor_idx] = 1
                        box_signal[jy, ix, anchor_idx] = 0

        for bbox_idx in range(num_bbox):
            # set anchor box each object
            cur_jy, cur_ix, cur_anchor_idx = best_anchor_for_bbox[bbox_idx]
            box_valid[cur_jy, cur_ix, cur_anchor_idx] = 1
            box_signal[cur_jy, cur_ix, cur_anchor_idx] = 1

            box_rpn_valid[cur_jy, cur_ix, 4 * cur_anchor_idx: 4 * cur_anchor_idx + 4] = 1
            box_rpn_reg[cur_jy, cur_ix, 4 * cur_anchor_idx: 4 * cur_anchor_idx + 4] = best_dx_for_bbox[bbox_idx]

            box_class[cur_jy, cur_ix, cur_anchor_idx] = labels[bbox_idx]['category']

        rpn_cls = np.concatenate([box_valid, box_signal], axis=2)
        rpn_reg = np.concatenate([box_rpn_valid, box_rpn_reg], axis=2)

        rpn_cls = np.expand_dims(rpn_cls, axis=0)
        rpn_reg = np.expand_dims(rpn_reg, axis=0)
        rpn_cls_valid = sum(sum(sum(box_signal)))
        return rpn_cls, rpn_reg, box_class, box_raw, rpn_cls_valid

    def select_highest_pred_box(self, pre_signal, pre_rpn_reg):
        rpn_width, rpn_height, num_anchors = pre_signal.shape[:3]
        num_all_anchors = rpn_width * rpn_height * num_anchors
        all_box = np.zeros((num_all_anchors, 4))  # coordinates of bounding boxes for each anchor box
        all_prob = np.zeros(num_all_anchors)  # probability of anchor box positive
        for ix in range(rpn_width):
            for jy in range(rpn_height):
                for anchor_idx in range(num_anchors):
                    # convert each predicted anchor regression to bounding boxes
                    cur_idx = ix * (rpn_height * num_anchors) + jy * num_anchors + anchor_idx
                    anchor_scale, anchor_ratio = self.anchor_sets[anchor_idx]

                    anchor_width = anchor_scale * anchor_ratio[0]
                    anchor_height = anchor_scale * anchor_ratio[1]

                    x1 = (ix + 0.5) * self.downscale - anchor_width / 2
                    y1 = (jy + 0.5) * self.downscale - anchor_height / 2

                    x2 = x1 + anchor_width
                    y2 = y1 + anchor_height

                    # convert predicted regression to bounding boxes
                    pred_box = inv_dx(pre_rpn_reg[ix, jy, 4 * anchor_idx: 4 * anchor_idx + 4], [x1, y1, x2, y2],
                                      self.img_shape[:2])
                    all_box[cur_idx, :4] = pred_box
                    all_prob[cur_idx] = pre_signal[ix, jy, anchor_idx]

        # Select boxes with highest probability
        selected = non_max_suppression_fast(all_box, all_prob)

        return selected

    def generate_classifier_data(self, selected, labels):
        anchor_boxes = selected[0]
        num_box = anchor_boxes.shape[0]

        anchor_cls = np.zeros((num_box, self.num_cls))
        anchor_reg_dx = np.zeros((num_box, 4 * (self.num_cls - 1)))
        anchor_reg_valid = np.zeros((num_box, 4 * (self.num_cls - 1)))
        anchor_pos = np.zeros(num_box)

        anchor_x = np.zeros((num_box, 4))
        best_iou = 0
        for anchor_box_idx in range(num_box):
            cur_anchor_box = anchor_boxes[anchor_box_idx]
            feature_box = self.bbox_to_fbox(cur_anchor_box)
            anchor_x[anchor_box_idx, :] = feature_box

            for label in labels:
                # calculate IoU for each bounding box with each object, determine the ground truth label of bounding box
                cur_gt_box = label['coordinates']

                cur_gt_cls = label['category']

                iou = calculate_iou(cur_anchor_box, cur_gt_box)
                if iou > best_iou:
                    best_iou = iou

                if iou > self.threshold:
                    anchor_cls[anchor_box_idx, cur_gt_cls] = 1
                    normalized_anchor = self.fbox_to_bbox(feature_box)
                    anchor_reg_valid[anchor_box_idx, 4 * cur_gt_cls - 4: 4 * cur_gt_cls] = 1
                    anchor_reg_dx[anchor_box_idx, 4 * cur_gt_cls - 4: 4 * cur_gt_cls] = cal_dx(cur_gt_box,
                                                                                               normalized_anchor)
                    anchor_pos[anchor_box_idx] = 1
                else:
                    anchor_cls[anchor_box_idx, 0] = 1
        anchor_reg = np.concatenate([anchor_reg_valid, anchor_reg_dx], axis=1)

        anchor_x = np.expand_dims(anchor_x, axis=0)
        anchor_cls = np.expand_dims(anchor_cls, axis=0)
        anchor_reg = np.expand_dims(anchor_reg, axis=0)

        return anchor_x, anchor_cls, anchor_reg, anchor_pos

    def generate_test_data(self, selected):
        anchor_boxes = selected[0]
        num_box = anchor_boxes.shape[0]
        anchor_x = np.zeros((num_box, 4))
        normalized_anchor = np.zeros((num_box, 4))
        for anchor_box_idx in range(num_box):
            cur_anchor_box = anchor_boxes[anchor_box_idx]
            feature_box = self.bbox_to_fbox(cur_anchor_box)
            anchor_x[anchor_box_idx, :] = feature_box
            normalized_anchor[anchor_box_idx, :] = self.fbox_to_bbox(feature_box)
        anchor_x = np.expand_dims(anchor_x, axis=0)
        return anchor_x, normalized_anchor

    def bbox_to_fbox(self, bbox):
        """
        convert bounding box to feature box
        @param bbox: (x1, y1, x2, y2) of bounding box
        @return: (x1, y1, w, h) of feature box
        """
        feature_box = [int(round(bbox[0] / self.downscale)), int(round(bbox[1] / self.downscale)),
                       int(round(bbox[2] / self.downscale)), int(round(bbox[3] / self.downscale))]
        fbox = cal_x1_and_length(feature_box)
        x1 = min(max(0, fbox[0]), self.feature_shape[1] - 2)
        y1 = min(max(0, fbox[1]), self.feature_shape[0] - 2)
        w = max(fbox[2], 2)
        h = max(fbox[3], 2)
        return x1, y1, w, h

    def fbox_to_bbox(self, fbox):
        """
        convert feature box to bounding box
        @param fbox: (x1, y1, w, h) of feature box
        @return:
        """
        x1 = fbox[0] * self.downscale
        y1 = fbox[1] * self.downscale
        x2 = x1 + fbox[2] * self.downscale
        y2 = y1 + fbox[3] * self.downscale
        return x1, y1, x2, y2
