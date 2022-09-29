# Can CNNs Count?
The purpose of this repo is to explore how good CNNs are at counting things. In `utils/dataloader.py` we define a PyTorch dataset class which creates black images with random numbers of white pixels.

The script `train.py` trains a simple LeNet CNN classifier on this counting dataset. It allows for the maximum and minimum number of white pixels to be varied, as well as the size of the input image. 
