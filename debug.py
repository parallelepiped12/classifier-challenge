#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:31:04 2022

@author: Thomas
"""

from utils import dataloader
import torch

dataset = dataloader.CountingDataset(0, 1, [10, 10], 5)
for i in torch.utils.data.DataLoader(dataset, batch_size=2):
    print(i[0])
