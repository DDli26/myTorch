import numpy as np
from resampling import *


class MaxPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        #to implement this, we must keep track of the index of the input element  which
        #was selected for a given element in the output of the pooling layer
        #these stored indices will help us in the backward pass,
        #since np.argmax returns a single index, we use np.unravel_index which
        #will give us the 2-d index we need

        #as for the forward pass, we use loops
        #i'm pretty sure it can be done with just 2 loops and numpy but since in pooling
        #the TA specifically mentioned that
        self.A=A
        batches, in_channels, self.in_h, self.in_w = A.shape
        kernel_len = self.kernel
        out_channels = in_channels
        out_w = self.in_w - kernel_len + 1
        out_h=self.in_h - kernel_len +1
        Z = np.zeros(shape=(batches, out_channels, out_w, out_h))
        self.pool_index = np.zeros_like(Z) #this will store the indices of the input, that was selected for a given element on Z

        for batch in range(batches):
            for channel in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        max_idx=np.argmax(A[batch, channel, h:h+kernel_len, w:w+kernel_len])
                        self.pool_index[batch, channel, h, w]= max_idx
                        r,c=np.unravel_index(max_idx, (kernel_len, kernel_len))
                        Z[batch, channel, h, w]=A[batch, channel, h+r, w+c]
        return Z



    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA=np.zeros_like(self.A)
        batches, out_channels, out_h, out_w=dLdZ.shape

        in_channels=out_channels
        for batch in range(batches):
            for channel in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        #now we find out which A index did the present Z entry come from
                        in_h, in_w = np.unravel_index(int(self.pool_index[batch, channel, h, w]), (self.kernel, self.kernel))
                        dLdA[batch, channel, h + in_h, w + in_w]+=dLdZ[batch, channel, h, w] #we use +=bcz the same value from A might contribute to multiple Z values
        return dLdA


# A=np.random.randint(1,10, size=(6,6))
# print(A, "\n")
# print(np.unravel_index(np.argmax(A), A.shape))


class MeanPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        #each Z entry is just the average of the elements of A that were under the
        #filter for that Z entry
        self.A=A
        batches, in_channels, in_h, in_w = A.shape
        kernel_size = self.kernel
        out_channels = in_channels
        out_h= in_h - kernel_size + 1
        out_w = in_w - kernel_size + 1
        Z=np.zeros(shape=(batches, out_channels, out_h, out_w))

        for h in range(out_h):
            for w in range(out_w):
                Z[:, :, h, w]= np.mean(A[:, :, h:h+kernel_size, w:w+kernel_size], axis=(2,3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA=np.zeros_like(self.A)
        kernel_size=self.kernel
        batches, out_channels, out_h, out_w = dLdZ.shape
        for h in range(out_h):
            for w in range(out_w):
                dLdA[:, :, h:h+kernel_size, w:w+kernel_size] += (dLdZ[:, :, h, w].reshape(batches, out_channels, 1,1)) #broadcasting will take care of dimensions

        dLdA= dLdA /(kernel_size * kernel_size) #finally taking the averages
        return dLdA

# A= np.random.randint(1,20, size=(3, 2, 2,2))
# print(f"{A}\n\n")
# print(np.mean(A, axis=(2,3)))

class MaxPool2d:
    """
    the generalized max pool class
    max pool with stride = x where x>1 is simply max-pool with stride 1
    followed by downsampling with a factor of x
    """
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        Z= self.maxpool2d_stride1.forward(A)
        Z= self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA= self.downsample2d.backward(dLdZ)
        dLdA= self.maxpool2d_stride1.backward(dLdA)

        return dLdA


class MeanPool2d:
    """
        the generalized mean pool class
        mean pool with stride = x where x>1 is simply mean-pool with stride 1
        followed by downsampling by factor of x
        """
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        return dLdA
