# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import random
import pickle
from torchvision import transforms
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


def load_lisa(database_path):

    train_path = os.path.join(database_path, 'train')
    test_path = os.path.join(database_path, 'test')
    train_img_paths = os.listdir(train_path)
    test_img_paths = os.listdir(test_path)

    train_data = np.zeros((len(train_img_paths), 32, 32, 3), dtype=np.uint8)
    train_labels = np.zeros((len(train_img_paths),), dtype=np.int)
    test_data = np.zeros((len(test_img_paths), 32, 32, 3), dtype=np.uint8)
    test_labels = np.zeros((len(test_img_paths),), dtype=np.int)

    for i, path in enumerate(train_img_paths):
        img_path = os.path.join(train_path, path)
        c, _ = map(int, path[:-4].split('_'))
        train_data[i] = cv2.imread(img_path)
        train_labels[i] = c

    for i, path in enumerate(test_img_paths):
        img_path = os.path.join(test_path, path)
        c, _ = map(int, path[:-4].split('_'))
        test_data[i] = cv2.imread(img_path)
        test_labels[i] = c

    return train_data, train_labels, test_data, test_labels


def load_gtsrb(database_path):

    train_path = os.path.join(database_path, 'train')
    test_path = os.path.join(database_path, 'test')
    train_img_paths = os.listdir(train_path)
    test_img_paths = os.listdir(test_path)

    train_data = np.zeros((len(train_img_paths), 32, 32, 3), dtype=np.uint8)
    train_labels = np.zeros((len(train_img_paths),), dtype=np.int)
    test_data = np.zeros((len(test_img_paths), 32, 32, 3), dtype=np.uint8)
    test_labels = np.zeros((len(test_img_paths),), dtype=np.int)

    for i, path in enumerate(train_img_paths):
        img_path = os.path.join(train_path, path)
        c, _ = map(int, path[:-4].split('_'))
        train_data[i] = cv2.imread(img_path)
        train_labels[i] = c

    for i, path in enumerate(test_img_paths):
        img_path = os.path.join(test_path, path)
        c, _ = map(int, path[:-4].split('_'))
        test_data[i] = cv2.imread(img_path)
        test_labels[i] = c

    return train_data, train_labels, test_data, test_labels




class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size()[0], n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss
