import numpy as np
from resampling import *


class Conv2d_stride1:
    """
    here we'll just extend what we did in Conv1d.py
    the einsum's used here are just an extension of the ones in Conv1d.py to 2-D channels
    I just blindly extended the string in the einsum and it worked. The important thing is
    that we understand how the forward pass and backward pass work
    """
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
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batches, in_channels, in_height, in_width = A.shape
        out_width = in_width - self.kernel_size + 1
        out_height=in_height-self.kernel_size +1
        Z = np.zeros(shape=(batches, self.out_channels, out_height,out_width))

        for i in range(out_height):
            for j in range(out_width):
                Z[:, :, i, j] = np.einsum(
                    "bchw, ochw->bo ",
                    A[:, :, i:i + self.kernel_size, j:j+self.kernel_size], self.W  # we ensure that width of A is the same as the kernel
                ) + self.b  # broadcasting will take care of bias, turning the row vector into a matrix

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, input_height, input_width = self.A.shape
        self.dLdW = np.zeros_like(self.W)

        conv_filter = dLdZ
        filter_height = dLdZ.shape[2]
        filter_width=dLdZ.shape[3]
        for i in range(input_height - filter_height + 1):
            for j in range(input_width-filter_width+1):
                self.dLdW[:, :, i,j] = np.einsum(
                    "bchw, bohw->oc",

                    self.A[:, :, i:i + filter_height, j: j + filter_width], conv_filter
                )


        self.dLdb = np.einsum("bohw->o", dLdZ)


        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0),
                                    (self.kernel_size - 1, self.kernel_size - 1), #along height of an output_map
                                    (self.kernel_size-1, self.kernel_size-1)) #along width of the map
                             )
        #flipped W is a 180 degree rotation of the W matrix, since each W channel is now 2D
        flipped_W = np.flip(self.W, axis=2)
        flipped_W=np.flip(flipped_W, axis=3) #flipping across height and then width, or vice versa gives the 180 degree rotatin
        dLdA = np.zeros_like(self.A)

        for i in range(padded_dLdZ.shape[2] - self.kernel_size + 1):
            for j in range(padded_dLdZ.shape[3]-self.kernel_size+1):
                dLdA[:, :, i, j] = np.einsum(
                    # b: batch, o:out_channels, k:kernel_size, c: in_channels
                    "bohw, ochw->bc",
                    padded_dLdZ[:, :, i:i + self.kernel_size, j:j+self.kernel_size], flipped_W
                )

        return dLdA



class Conv2d:
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

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A= np.pad(A, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)))
        #again, just like we did in Conv1d, convolving with a stride is same as
        #convolving and then downsampling
        Z=self.conv2d_stride1.forward(A)
        Z= self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdA=self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA=self.conv2d_stride1.backward(dLdA)

        # Unpad the gradient
        if self.pad!=0:
            dLdA= dLdA[:,:, self.pad:-self.pad, self.pad: -self.pad]
        return dLdA