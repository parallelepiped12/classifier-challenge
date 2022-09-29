#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:37:46 2022

@author: Thomas
"""
import torch
import random


class CountingDataset(torch.utils.data.IterableDataset):
    def __init__(self, min_objects, max_objects, image_size, object_size=1):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.image_size = image_size
        self.object_size = object_size

    def __iter__(self):
        return self

    def __next__(self):
        num_objects = random.randint(self.min_objects, self.max_objects)
        return make_torch_image(num_objects, self.image_size, self.object_size), num_objects


def make_torch_image(num_objects, image_size, object_size=1, indices=None):
    image = torch.zeros(1, image_size[0], image_size[1])
    num_object_places = (image_size[0] // object_size) * (image_size[1] // object_size)
    # Indices can be specified for validation and test datasets, otherwise
    # they are allocated randomly
    if indices is None:
        indices = random.sample(range(num_object_places), num_objects)
    indices = torch.tensor(indices)
    idx_h = torch.remainder(indices, image_size[0] // object_size) * object_size
    idx_w = indices.div(image_size[0] // object_size, rounding_mode="floor") * object_size
    idx = torch.stack([idx_h, idx_w])
    # this process of looping through the pixels to modify is too slow
    # the speed makes it infeasible to train models with a lot of classes
    # i think it could be done faster using scatter
    for i in range(num_objects):
        image[0, idx[0, i]:idx[0, i]+object_size, idx[1, i]:idx[1, i]+object_size] = 1
    return image


class FiniteCountingDataset(torch.utils.data.Dataset):
    def __init__(self, num_objects, indices, image_size, object_size=1):
        self.image_size = image_size
        self.object_size = object_size
        self.num_objects = num_objects
        self.indices = indices
        assert len(num_objects) == len(indices)

    def __len__(self):
        return len(self.num_objects)

    def __getitem__(self, idx):
        return make_torch_image(self.num_objects[idx], self.image_size,
                                self.object_size, self.indices[idx]), self.num_objects[idx]


def make_indices(num_objects, num_object_places):
    indices = []
    for n_obj, n_obj_places in zip(num_objects, num_object_places):
        indices.append(torch.randperm(n_obj_places)[:n_obj])
    return indices
