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
            ) + self.b #broadcasting will take care of bias, turning the row vector into a matrix

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #remember, derivative wrt a given parameter is the average of derivatives for the batch_size
        #the division by the batch size is carried over in the backward pass from dLdZ, we have to take care of the
        #summations
        self.dLdW =  np.zeros_like(self.W) # shape: out_channels x in_channels x kernel_size
        #dLdW wrt ith channel of  is convolution between ith input channel and corrsponding output channel
        #if done over the batch, derivative wrt ith channel is summations of the results of individual convolutions
        conv_filter=dLdZ
        filter_size=dLdZ.shape[2]
        for i in range(self.A.shape[2]-filter_size+1):
            #b:batches, c:in_channels, k: filter_size as computed above, o:out_channels
            self.dLdW[:,:,i]=np.einsum( #derivate of loss wrt ith entry of all channels of all filters
                "bck, bok->oc", #an easy way to understand is to fix o and c to some index, so o and c mean oth filters cth channel
                                         #so ith entry of oth filter's cth channel is calculated by placing dLdZ[b, o, :] on self.A[b,c] and then
                                         #performing elementwise multiplication and adding the result along all batches since
                                         # the addition along all batches accounts for the fact that derivative wrt a given parameter
                                        # is an average of the derivatives wrt individual inputs
                self.A[:,:,i:i+filter_size], dLdZ
            )

        #dLdb
        #for a single input, dLdb wrt bias of jth filter  is just the sum of the entries of jth output_channel
        #to do this for the entire batch, we'd have to sum the bias for jth filter over all batches
        #and then divide by #batches. well division by #batches is carried over in the backward pass
        #through loss layer, so we just have to take care of the summation
        self.dLdb =  np.einsum("bok->o", dLdZ)

        #finding dLdA:
        #firstly, each channel of each input, affects all outputs
        #derivative wrt jth channel of each input is a convolutions between the flipped
        # W's corresponding channel, that is, jth channel of all filters: shape(out_channels, j, kernel_size);
        # and padded dLdZ
        #we have to do this for every input channel of every batch
        #the for loop below moves the filter over dLdZ and einsum gives us the ith entry of
        #all channels of dLdA for all batches, thus computing dLdA in one pass over dLdZ
        padded_dLdZ= np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1,self.kernel_size-1)))
        flipped_W=np.flip(self.W, axis=2)
        dLdA= np.zeros_like(self.A)


        for i in range(padded_dLdZ.shape[2]-self.kernel_size+1):
            dLdA[:, :, i]= np.einsum(
                #b: batch, o:out_channels, k:kernel_size, c: in_channels
                "bok, ock->bc",
                padded_dLdZ[:,:,i:i+self.kernel_size], flipped_W
            )

        return dLdA

# z=np.random.randint(1,10, size=(3,3,5))
# #if kernel size if k, we'll have to pad each output_size with k-1
# print(z)
# padded_z= np.pad(z,((0,0), (0,0), (2,2)) )
# print(padded_z)
# W= np.random.randint(1,5, size=(2,3,6))
# print(f"W: {W}")
# flipped_W= np.flip(W, axis=2)
# print(f"\n\nflipped_W: {flipped_W}")
class Conv1d:
    """
    to implement a Conv1d with a given stride, what we can do is first perform a Conv1d with stride 1,
    as seen in Conv1d_stride1d and them perform downsampling with a factor of the given stride
    """
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

        # Initialize Conv1d() and Downsample1d() instance

        self.conv1d_stride1 =  Conv1d_stride1(in_channels,
                                              out_channels,
                                              kernel_size,
                                              weight_init_fn,
                                              bias_init_fn)
        self.downsample1d =  Downsample1d(downsampling_factor=self.stride)


    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """


        #firstly we might wanna perform padding on A, as needed
        A=np.pad(A,((0,0), (0,0), (self.pad, self.pad)))
        # Call Conv1d_stride1 to obtain the colvolution output
        Z=self.conv1d_stride1.forward(A)

        # downsample, this performs the work of striding
        #so the final output is what we would get by performing convolution with a give stride
        Z=self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #so backward pass is simply performing backward pass via the downsampled layer
        #and then via the convolution layer
        #now the dLdA we get this way will be be of the same dimesnions as the padded A
        #which is not the original dimension of A
        #so we must remove the padding for the computed dLdA
        # Call downsample1d backward
        dLdA= self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA= self.conv1d_stride1.backward(dLdA)

        # Remove padding
        if self.pad!=0:
            dLdA= dLdA[:, :, self.pad:-self.pad]

        return dLdA
