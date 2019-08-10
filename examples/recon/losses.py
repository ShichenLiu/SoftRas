import torch
import torch.nn as nn
import numpy as np


def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

def iou_loss(predict, target):
    return 1 - iou(predict, target)

def multiview_iou_loss(predicts, targets_a, targets_b):
    loss = (iou_loss(predicts[0][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[1][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[2][:, 3], targets_b[:, 3]) + \
            iou_loss(predicts[3][:, 3], targets_b[:, 3])) / 4
    return loss