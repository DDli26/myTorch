# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys


sys.path.append("mytorch")


class CNN_SimpleScanningMLP:
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...

        # <---------------------
        # 24x128 is dimension of input a single input
        # so 24 is the no. of channels, 128 is the input width, bcz input is 1-d
        #our kernel will slide over these 128 timesteps.
        #the scanning mlp has 8 neurons in first layer, 16 in second and 4 in third
        #the no. of neurons equals the no. of kernels, since the weights of a neuron are
        #nothing but the weight of the kernel and output maps for a layer also equals the no. of
        #neurons, one output map per neuron
        #kernel size is 8
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16,4, 1, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        #now these weights are the weights of the kernels/filters
        #we have to get an idea of the shape of the weights
        #and based on that decide which weight is for which kernel
        # and initialize the weights of each convolutions layer: out_channel x  in_channel x kernel_size
        # print(f"Simple scanning MLP:\n shape of w1 {w1.shape},\n shape of w2 {w2.shape} \n shape of w3 {w3.shape}")
        # w1(192, 8),
        # w2(8, 16)
        # w3(16, 4)
        self.conv1.conv1d_stride1.W = w1.T.reshape(8,8,24).transpose(0, 2,1)
        self.conv2.conv1d_stride1.W = w2.T.reshape(16, 1, 8).transpose(0,2,1)
        self.conv3.conv1d_stride1.W = w3.T.reshape(4, 1, 16).transpose(0,2,1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        print(f"Cnn_simpleScanningMLP. Input shape is : {A.shape}")
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP:
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels = 24, out_channels = 2, kernel_size= 2, stride = 2)
        self.conv2= Conv1d(2, 8, 2,2)
        self.conv3= Conv1d(8, 4, 2, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        w1, w2, w3 = weights
        print(f"Distributed Scanning MLP:\nshape of w1 {w1.shape},\n shape of w2 {w2.shape} \n shape of w3 {w3.shape}")

        # w1(192, 8),
        # w2(8, 16)
        # w3(16, 4)
        #slicing cause of the shared weights
        w1= w1[:48, :2]
        w2= w2[ :4, :8]
        w3= w3[ :16, :4]
        # out channels, kernel size, in channels
        self.conv1.conv1d_stride1.W = w1.T.reshape(2,2,24).transpose(0,2,1)
        self.conv2.conv1d_stride1.W = w2.T.reshape(8, 2,2).transpose(0,2,1)
        self.conv3.conv1d_stride1.W = w3.T.reshape(4, 2, 8).transpose(0,2,1)


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
