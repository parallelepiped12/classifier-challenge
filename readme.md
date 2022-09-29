# Can CNNs Count?
The purpose of this repo is to explore how good CNNs are at counting things. In `utils/dataloader.py` we define a PyTorch dataset class which creates black images with random numbers of white pixels.

The script `train.py` trains a simple LeNet CNN classifier on this counting dataset. It allows for the maximum and minimum number of white pixels to be varied, as well as the size of the input image. 

Using a large number of classes is somewhat infeasible with the current implementation due to its speed, so we have only tested with 11 classes (0-10 pixels added to the image). In this setting we were able to achieve 98% validation accuracy without tuing the training parameters at all.

The script `debug.py` prints the output of the dataloader. I used this in lieu of visualisation since my images are small and all values are 0 and 1.

To allow fixed validation and test datasets to be defined I have created a finite version of the (normally infinite) dataset. The random parameters are fixed in advance and saved.

The best trained model can be found in `models/initial/100.pt`

I used Python 3.8.10, and the only library I used was PyTorch 1.12.1. Earlier versions of PyTorch will likely work fine.
