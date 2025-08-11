import numpy as np


class Flatten:

    #input in real cases is of the shape b, in_c, in_h, in_w
    #I believe our code works regardless
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A=A
        print(f"flatten layer input shape {A.shape}")
        Z=  A.reshape(A.shape[0], -1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        # batches, in_channels, in
        dLdA= dLdZ.reshape(*self.A.shape)
        return dLdA
