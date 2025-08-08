# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A=A
        batches, in_channels, in_size=A.shape
        out_width=in_size-self.kernel_size+1
        Z=np.zeros(shape=(batches, self.out_channels, out_width))
        for i in range(out_width):
            # W.shape: (out_channels, in_channels, kernel_size)
            #b: batch, c:in_channels, k: kernel_size, o: out_channels
            #verify that this einsum will output the ith index output needed
            # for each batch and for each output_channel
            Z[:,:,i]=np.einsum(
                "bck, ock->bo ",
                A[:,:,i:i+self.kernel_size], self.W #we ensure that width of A is the same as the kernel
            ) + self.b #broadcasting will take care of bias

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # self.dLdW =  # TODO
        # self.dLdb =  # TODO

        #dLdA: for a single channel of a single input is a convolution
        #between padded dLdZ of a single output channel and flipped filter of that channel
        #do the appropriate padding on each output channels of each batch
        padded_dLdZ= np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1,self.kernel_size-1)))
        flipped_W=np.flip(self.W, axis=2)
        dLdA= np.zeros_like(self.A)
        for i in range(padded_dLdZ.shape[2]-self.kernel_size+1):
            dLdA[:, :, i]= np.einsum(
                #wait, we aren't accounting for the fact that one input channel affects all output channels
                "",
                padded_dLdZ[:,:,i+self.kernel_size], flipped_W
            )

        return None

# z=np.random.randint(1,10, size=(3,3,5))
# #if kernel size if k, we'll have to pad each output_size with k-1
# print(z)
# padded_z= np.pad(z,((0,0), (0,0), (2,2)) )
# print(padded_z)
W= np.random.randint(1,5, size=(2,3,6))
print(f"W: {W}")
flipped_W= np.flip(W, axis=2)
print(f"\n\nflipped_W: {flipped_W}")
class Conv1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)))

        # Call Conv1d_stride1
        stride = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(stride)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        # Remove padding
        if self.pad != 0:
            dLdA = dLdA[:, :, self.pad : -self.pad]

        return dLdA
