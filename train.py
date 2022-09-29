#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:26:12 2022

@author: Thomas
"""
from utils import dataloader
from utils import networks
import torch
import itertools
num_epochs = 100
batches_per_epoch = 1000 # dataloader is infinite so there are no real epochs
batch_size = 100
image_size = [32, 32]
min_objects = 0
max_objects = 10

network = networks.LeNet(num_classes=max_objects-min_objects+1)
network.train()
optimiser = torch.optim.SGD(network.parameters(), lr=0.01)
loss_function = torch.nn.CrossEntropyLoss()

dataset = dataloader.CountingDataset(min_objects, max_objects, image_size)
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    dataloader = itertools.islice(torch.utils.data.DataLoader(dataset, batch_size=batch_size), 0, batches_per_epoch)
    print(f'-------{epoch}--------')
    for i, (images, classes) in enumerate(dataloader):
        optimiser.zero_grad()
        output = network(images)
        loss = loss_function(output, classes)
        loss.backward()
        optimiser.step()
        epoch_loss += loss / batches_per_epoch
        epoch_accuracy += torch.sum(classes == torch.argmax(output, dim=1)) / (batches_per_epoch * batch_size)
    torch.save(network.state_dict(), f'models/initial/{epoch+1}.pt')
    print(epoch_loss, epoch_accuracy)
