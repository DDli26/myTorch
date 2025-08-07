import math
import numpy as np


class Upsample1d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # implement Z
        #out_width = upsampling_factor * (in_width - 1)+1
        #input of size 5 will become of size 9 with a factor of 2, input size 6 will become 11
        Z=np.zeros(shape=(A.shape[0], A.shape[1], self.upsampling_factor * (A.shape[2]-1)+1))

        input_idx=0
        output_idx=0
        while output_idx<Z.shape[2]:
            Z[:, :, output_idx] = A[:, :, input_idx]
            input_idx+=1
            output_idx+=self.upsampling_factor

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        """
            backward pass of upsampling is simply downsampling dLdZ.
            we do it for the entire batch and all channels of each input of the batch
            out_width = upsampling_factor * (in_width - 1)+1
        """
        input_width= (dLdZ.shape[2]-1) // self.upsampling_factor +1
        dLdA=np.zeros(shape=(dLdZ.shape[0], dLdZ.shape[1], input_width))
        output_idx=0
        for input_idx in range(input_width):
            dLdA[:, :, input_idx]= dLdZ[:, :, output_idx]
            output_idx+=self.upsampling_factor

        return dLdA

#testing forward
# A= np.array([1,0,-1,2,1]).reshape(1,1,5)
#
# up_layer=Upsample1d(2)
# print(up_layer.forward(A))

class Downsample1d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        N, C, self.Win = A.shape
        Wout = math.ceil(self.Win / self.downsampling_factor)
        Z = np.zeros((N, C, Wout))

        for i in range(Wout):
            if i * self.downsampling_factor < self.Win:
                Z[:, :, i] = A[:, :, i * self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        N, C, Wout = dLdZ.shape
        dLdA = np.zeros((N, C, self.Win))

        for i in range(Wout):
            if i * self.downsampling_factor < self.Win:
                dLdA[:, :, i * self.downsampling_factor] = dLdZ[:, :, i]

        return dLdA


class Upsample2d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, Hin, Win = A.shape
        Hout = self.upsampling_factor * (Hin - 1) + 1
        Wout = self.upsampling_factor * (Win - 1) + 1

        Z = np.zeros((N, C, Hout, Wout))

        for i in range(Hin):
            for j in range(Win):
                Z[:, :, i * self.upsampling_factor, j * self.upsampling_factor] = A[
                    :, :, i, j
                ]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape
        Hin = (Hout - 1) // self.upsampling_factor + 1
        Win = (Wout - 1) // self.upsampling_factor + 1

        dLdA = np.zeros((N, C, Hin, Win))

        for i in range(Hin):
            for j in range(Win):
                dLdA[:, :, i, j] = dLdZ[
                    :, :, i * self.upsampling_factor, j * self.upsampling_factor
                ]

        return dLdA


class Downsample2d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, self.Hin, self.Win = A.shape
        Hout = math.ceil(self.Hin / self.downsampling_factor)
        Wout = math.ceil(self.Win / self.downsampling_factor)

        Z = np.zeros((N, C, Hout, Wout))

        for i in range(Hout):
            for j in range(Wout):
                if (
                    i * self.downsampling_factor < self.Hin
                    and j * self.downsampling_factor < self.Win
                ):
                    Z[:, :, i, j] = A[
                        :, :, i * self.downsampling_factor, j * self.downsampling_factor
                    ]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape

        dLdA = np.zeros((N, C, self.Hin, self.Win))

        for i in range(Hout):
            for j in range(Wout):
                dLdA[
                    :, :, i * self.downsampling_factor, j * self.downsampling_factor
                ] = dLdZ[:, :, i, j]

        return dLdA
