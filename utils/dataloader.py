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
    if indices is None:
        indices = torch.randperm(num_object_places)[:num_objects]
    for i in indices:
        idx_h = (i % (image_size[0] // object_size)) * object_size
        idx_w = (i // (image_size[0] // object_size)) * object_size
        image[0, idx_h:idx_h+object_size, idx_w:idx_w+object_size] = 1
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
                                self.object_size, self.incides[idx])


def make_indices(num_objects, num_object_places):
    indices = []
    for n_obj, n_obj_places in zip(num_objects, num_object_places):
        indices.append(torch.randperm(n_obj_places)[:n_obj])
    return
