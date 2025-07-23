import numpy as np
from activation import Softmax

"""
            if you remember from mlp.py the models that we defined only has normal layers
            the final layer where loss is calculated was not defined.
            now the ouputs of the mlps gets passed through a loss layer
            and backward pass also starts from the loss layer
            also notice that when working with CrossEntropyLoss, the softmax activation
            is included inside the forward pass. This makes our task much more convenient.
            In the sense that the backward passes of the loss layers are coded so that they
            return the derivatives wrt the ouputs of the mlps, that is the output before softmax
            activation. 
            As for MSE, no activation is necessary

    """
class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """
        #so we'll have to calculate row-wise msc, and MSE will be a vector of dimension Nx1
        #where N is batch size
        self.A=A #we'll need this for the backward pass
        self.Y=Y
        self.N= A.shape[0]
        self.C= A.shape[1]

        #see the output of the network can be C regression terms, that is why we have
        # A and Y of dimension NxC, where N is the batch size
        # so to find the divergence for an individual input, we have to sum the square
        # differences of C terms (that's the l2 norm of the vector) and divide by C
        # loss is obviously the average of the divergences over
        # all the inputs, so we divide all the divergences by N
        square_error= np.square(self.A - self.Y)
        divergences= np.sum(square_error, axis=1) / self.C
        self.L= np.sum(divergences, axis=0)/self.N

        return self.L


    def backward(self):
        # we must return the derivative wrt the final output of the network
        # since the final output for a batch is NxC
        # the derivative will also be of the same dimension
        # the denominator is carried over in backprop, ensuring that the derivatives
        # for the weights and biases are also the average derivatives over all input of the batch
        dLdA =  ( 2 * (self.A - self.Y) ) / (self.N * self.C)
        return dLdA




class CrossEntropyLoss:
    """
        if you remember from mlp.py the models that we defined only has normal layers
        the final layer where loss is calculated was not defined.
        now the ouputs of the mlps gets passed through a loss layer
        and backward pass also starts from the loss layer
        also notice that when working with CrossEntropyLoss, the softmax activation
        is included inside the forward pass. This makes our task much more convenient.
        In the sense that the backward passes of the loss layers are coded so that they
        return the derivatives wrt the ouputs of the mlps, that is the output before softmax
        activation

    """
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.Y=Y
        self.N= A.shape[0]
        self.C= A.shape[1]
        #softmax activation
        self.A= Softmax().forward(A)
        self.pred_probs_of_correct_out= np.sum((self.A * self.Y), axis=1) # this is a 1-d array (dim of N) of  probabilities for the correct label
        #loss is the average of the divergences
        divergences= -np.log(self.pred_probs_of_correct_out)
        self.L = np.sum(divergences)/self.N
        return self.L

    def backward(self):
        pass


