import cv2
from helper.roi_helper import *


class Drawer:
    def __init__(self, raw_img, cur_size, output_dir):
        self.raw_img = raw_img
        self.cur_size = cur_size[:2]
        self.raw_size = raw_img.shape[:2]
        self.output_dir = output_dir
        self.zoom = (self.raw_size[1] / cur_size[1], self.raw_size[0] / cur_size[0])

    def draw_labels(self, labels, color=(0, 255, 0)):
        for i in range(len(labels)):
            point_1 = (int(self.zoom[0] * labels[i]['coordinates'][0]), int(self.zoom[1] * labels[i]['coordinates'][1]))
            point_2 = (int(self.zoom[0] * labels[i]['coordinates'][2]), int(self.zoom[1] * labels[i]['coordinates'][3]))
            cv2.rectangle(self.raw_img, point_1, point_2, (0, 255, 0))

    def draw_gt_labels(self, rpn_cls, rpn_reg, raw, color=(0, 0, 255)):
        num_of_anchors = rpn_cls.shape[3] // 2
        valid = rpn_cls[0, :, :, :num_of_anchors]
        signal = rpn_cls[0, :, :, num_of_anchors:]
        for ix in range(valid.shape[0]):
            for jy in range(valid.shape[1]):
                for idx in range(valid.shape[2]):
                    if signal[ix, jy, idx] == 1:
                        raw_box = raw[ix, jy, 4 * idx: 4 * idx + 4]
                        reg_box = rpn_reg[0, ix, jy, 4 * num_of_anchors + 4 * idx: 4 * num_of_anchors + 4 * idx + 4]
                        pre_box = inv_dx(reg_box, raw_box, self.cur_size)
                        pre_box_in_raw = [
                            int(pre_box[0] * self.zoom[0]), int(pre_box[1] * self.zoom[1]),
                            int(pre_box[2] * self.zoom[0]), int(pre_box[3] * self.zoom[1])
                        ]
                        cv2.rectangle(self.raw_img, (pre_box_in_raw[0], pre_box_in_raw[1]),
                                      (pre_box_in_raw[2], pre_box_in_raw[3]), (0, 0, 255))

    def paint(self):
        cv2.imwrite(self.output_dir, self.raw_img)
