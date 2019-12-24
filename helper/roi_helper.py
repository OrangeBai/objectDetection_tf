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


def cal_center_and_length(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return cx, cy, w, h


def cal_x1_and_length(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return x1, y1, w, h


def cal_dx(gt_box, anchor_box):
    cx_gt, cy_gt, w_gt, h_gt = cal_center_and_length(gt_box)

    cx_anchor, cy_anchor, w_anchor, h_anchor = cal_center_and_length(anchor_box)

    tx = (cx_gt - cx_anchor) / w_anchor
    ty = (cy_gt - cy_gt) / h_anchor
    tw = np.log((w_gt / w_anchor))
    th = np.log((h_gt / h_anchor))

    return tx, ty, tw, th


def inv_dx(dx, anchor_box, img_shape):
    """

    @param dx: (tx, ty, tw, th)
    @param anchor_box: (x1, y1, x2, y2)
    @return:
    """
    cx_anchor, cy_anchor, w_anchor, h_anchor = cal_center_and_length(anchor_box)

    w_pred = np.exp(dx[2]) * w_anchor
    h_pred = np.exp(dx[3]) * h_anchor

    cx_pred = dx[0] * w_anchor + cx_anchor
    cy_pred = dx[1] * h_anchor + cy_anchor

    x1_pre = cx_pred - w_pred / 2.0
    x2_pre = cx_pred + w_pred / 2.0
    y1_pre = cy_pred - h_pred / 2.0
    y2_pre = cy_pred + h_pred / 2.0

    return clip_box([x1_pre, y1_pre, x2_pre, y2_pre], img_shape)


def non_max_suppression_fast(boxes, prob, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(prob)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    prob = prob[pick]

    return boxes, prob


def bbox_to_fbox(bbox, downscale):
    x1 = int(round(bbox[0] / downscale))
    y1 = int(round(bbox[1] / downscale))
    x2 = int(round(bbox[2] / downscale))
    y2 = int(round(bbox[3] / downscale))
    return x1, y1, x2, y2


def fbox_to_bbox(fbox, downscale):
    x1 = fbox[0] * downscale
    y1 = fbox[1] * downscale
    x2 = fbox[2] * downscale
    y2 = fbox[3] * downscale
    return x1, y1, x2, y2


def clip_box(box, img_shape):
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    x2 = min(box[2], img_shape[1])
    y2 = min(box[3], img_shape[0])
    return x1, y1, x2, y2


def clip_fbox(fbox, feature_shape):
    x1 = min(max(0, fbox[0]), feature_shape[1] - 2)
    y1 = min(max(0, fbox[1]), feature_shape[0] - 2)
    w = max(fbox[2], 2)
    h = max(fbox[3], 2)
    return x1, y1, w, h