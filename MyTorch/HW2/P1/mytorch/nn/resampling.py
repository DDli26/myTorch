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
        self.input_width=A.shape[2]
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
            using 
            out_width = upsampling_factor * (in_width - 1)+1 
            to calculate input_width
        """
        #width of the input to the layer in forward pass:

        dLdA=np.zeros(shape=(dLdZ.shape[0], dLdZ.shape[1], self.input_width))
        output_idx=0
        for input_idx in range(self.input_width):
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
        batch_size, in_channels, self.in_width=A.shape
        #out_width can be determined by using the inverse of the formula used in Upsample1d
        out_width = (self.in_width - 1) // self.downsampling_factor + 1
        Z=np.zeros(shape=(batch_size, in_channels, out_width))
        in_idx=0
        # print(f"input width: {in_width}, output width: {out_width}")
        for i in range(out_width):
            # prnt(f"")
            Z[:, :, i]= A[:, : , in_idx]
            in_idx+=self.downsampling_factor


        return Z


    def backward(self, dLdZ):
        """
        backward of downsampling is upsampling, but the derivative of loss
        wrt the values that are lost during forward pass, as a result of downsampling,
        is zero
        """
        batch_size, channels, out_width=dLdZ.shape
        dLdA= np.zeros(shape=(batch_size, channels, self.in_width))
        dLdA_idx=0
        for i in range(out_width):
            dLdA[:, :, dLdA_idx]=dLdZ[:, :, i]
            dLdA_idx+=self.downsampling_factor #so we skip the appropariate no. of entries
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

        to determine the output_height, and output_width we can use the same formula we used in Upsample1D's forward pass
        out_width = upsampling_factor * (in_width - 1)+1
        """
        batch_size, in_channels, self.in_height, self.in_width=A.shape

        out_height= self.upsampling_factor * (self.in_height-1) +1
        out_width= self.upsampling_factor * (self.in_width-1) +1

        Z=np.zeros(shape=(batch_size, in_channels, out_height, out_width))
        out_height_idx =0
        for i in range(self.in_height):
            out_width_idx=0
            for j in range(self.in_width):
                Z[:, : ,out_height_idx, out_width_idx]=A[:, :, i, j]
                out_width_idx+=self.upsampling_factor

            out_height_idx+=self.upsampling_factor

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)

        backward pas of upsampling is simply downsampling
        """
        batch_size, in_channels, out_height, out_width=dLdZ.shape
        dLdA= np.zeros(shape=(batch_size, in_channels, self.in_height, self.in_width))

        out_height_idx=0
        for h in range(self.in_height):
            out_width_idx=0
            for w in range(self.in_width):
                dLdA[:, :, h, w]=dLdZ[:, :, out_height_idx, out_width_idx]
                out_width_idx+=self.upsampling_factor

            out_height_idx+=self.upsampling_factor


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
        batch_size, in_channels, self.input_height, self.input_width=A.shape

        # we can use the inverse of the formulae used in Upsample2D to determine, output_height, and width
        #out_width = upsampling_factor * (in_width - 1)+1
        output_height=  (self.input_height-1) // self.downsampling_factor +1
        output_width= (self.input_width-1) // self.downsampling_factor +1

        Z= np.zeros(shape=(batch_size, in_channels, output_height, output_width))
        input_height_idx=0
        for h in range(output_height):
            input_width_idx=0
            for w in range(output_width):
                Z[:,:, h, w]= A[:, :, input_height_idx, input_width_idx]
                input_width_idx+=self.downsampling_factor

            input_height_idx+=self.downsampling_factor
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)

        backward pass in downsampling is upsampling
        """
        batch_size, channels, output_height, output_width=dLdZ.shape
        dLdA=np.zeros(shape=(batch_size, channels, self.input_height, self.input_width))

        in_height_idx=0
        for h in range(output_height):
            in_width_idx=0
            for w in range(output_width):
                dLdA[:, :, in_height_idx, in_width_idx]= dLdZ[:, :, h, w]
                in_width_idx+=self.downsampling_factor
            in_height_idx+=self.downsampling_factor

        return dLdA
