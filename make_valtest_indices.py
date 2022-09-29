#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:38:09 2022

@author: Thomas
"""
import torch
from utils import dataloader
from itertools import chain

try:
    torch.load('valtest_indices/val_image_size_32_object_size_1_num_objects_10.pt')
except FileNotFoundError:
    num_objects = []
    for i in range(11):
        num_objects += ([i] * 100)
    num_object_places = [32 * 32] * len(num_objects)
    indices = dataloader.make_indices(num_objects, num_object_places)
    torch.save(indices, 'valtest_indices/val_image_size_32_object_size_1_num_objects_10.pt')
