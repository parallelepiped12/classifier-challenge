#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:37:46 2022

@author: Thomas
"""
import torch
import random


class CountingDataset(torch.utils.data.IterableDataset):
    def __init__(self, min_objects, max_objects, image_size):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.image_size = image_size

    def __iter__(self):
        return self

    def __next__(self):
        num_objects = random.randint(self.min_objects, self.max_objects)
        return make_torch_image(num_objects, self.image_size), num_objects


def make_torch_image(num_objects, image_size):
    image = torch.zeros(1, image_size[0], image_size[1])
    indices = torch.randperm(image_size[0]* image_size[1])[:num_objects]
    for i in indices:
        image[0, i % image_size[0], i // image_size[0]] = 1
    return image
