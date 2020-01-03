import cv2
import numpy as np

def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def union(a, b, area_intersection):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def calculate_iou(a, b):
    """

    @param a: ground truth box coordinates: ((x1, y1), (x2, y2))
    @param b: anchor box coordinates: ((x1, y1), (x2, y2))
    @return: IoU
    """
    for box, i in [(box, i) for box in [a, b] for i in [0, 1]]:
        if box[i] >= box[i + 2]:
            return 0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

def cal_mvp(pred_all, gt_all, threshold):
    """
    THis function is used for calculate mAP of object detection project.
    @param pred: predicted value:
                [
                    [pic_num, category, probability, x1, y1, x2, y2],
                    [pic_num, category, probability, x1, y1, x2, y2],
                    ...
                    [pic_num, category, probability, x1, y1, x2, y2]
                ]
    @param gt: ground truth value:
                [
                    [pic_num, category, x1, y1, x2, y2],
                    [pic_num, category, x1, y1, x2, y2],
                    ...
                    [pic_num, category, x1, y1, x2, y2]
                ]
    @param threshold: threshold of IoU to set a positive tag:
    @return:
    """
    num_pre = len(pred_all)
    num_gt = len(gt_all)
    pred_tags = np.zeros((num_pre,3))
    gt_found = np.zeros((num_gt, 4))         # store the current status of gt objects, (found, pre_idx, best_iou, prob)
    for cur_gt_idx in range(len(gt_all)):
        cur_gt = gt_all[cur_gt_idx]
        for cur_pred_idx in range(len(pred_all)):
            cur_pred = pred_all[cur_pred_idx]
            if cur_pred[0] != cur_gt[0]:
                continue                    # if not the same pic, continue
            if cur_pred[1] != cur_gt[1]:        # if category not match, continue
                continue
            cur_IoU = calculate_iou(cur_pred[3:], cur_gt[2:])
            if cur_IoU < threshold:
                # if current IoU less than threshold, the cur_pred is negative to cur_gt
                continue
            else:
                # if current IoU is greater than threshold
                if not gt_found[cur_gt_idx, 0]:
                    # if it is the first positive for the gt object, set labels of the ground truth
                    gt_found[cur_gt_idx] = [1, cur_pred_idx, cur_IoU, cur_pred[2]]
                if gt_found[cur_gt_idx, 0]:
                    # if it is not the first positive for the gt object, the set the higher one TP and lower one FP
                    if cur_IoU < gt_found[cur_gt_idx, 2]:
                        pred_tags[cur_pred_idx, 0] = 0
                    else:
                        pred_tags[cur_pred_idx, 0] = 1
                        prev_idx = gt_found[1]
                        pred_tags[prev_idx, 0] = 0
                        gt_found[cur_gt_idx] = [1, cur_pred_idx, cur_IoU, cur_pred[2]]

    for cur_pred_idx in range(len(pred_all)):
        pred_tags[cur_pred_idx] = pred_all[1:3]



