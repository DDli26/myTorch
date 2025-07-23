import numpy as np
from mytorch.nn.activation import Softmax

"""
            if you remember from mlp.py the models that we defined only computed the 
            outputs of the network. or in case of classification, only the logits
            after those outputs, if it is a regression task, we can simply  use forward pass
            of MSELoss. for backprop, we can start with backward function os MSELoss, which returns
            the derivative wrt the network outputs. 
            
            for Cross-entropy, things are a little different. our models in mlp.py output the logits.
            given the logits, we then pass them through a softmax layer and finally calculate the loss. 
            all this happens in the forward pass of the CrossEntropyLoss class
            the backward pass of CrossEntropy loss return the derivative of the Loss wrt the logits (input to softmax)
            which is a really simply formula, so we don't go through the trouble of invoking the backward
            pass for the Softmax.  

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
        # A and Y of dimension NxC, where N is the batch size and C is the no. of outputs
        # so to find the divergence for an individual input, we have to sum the square
        # differences of C terms (that's the l2 norm of the output vector) and divide by C
        # loss is obviously the average of the divergences over
        # all the inputs, so we divide the resulting divergences by N
        square_error= np.square(self.A - self.Y)
        divergences= np.sum(square_error, axis=1) / self.C # Nx1 --> divergences for each of the N inputs
        self.L= np.sum(divergences, axis=0)/self.N

        return self.L


    def backward(self):
        # we must return the derivative wrt the final output of the network
        # since the final output for a batch is NxC
        # the derivative will also be of the same dimension
        # the denominator is carried over in backprop, ensuring that the derivatives
        # for the weights and biases are also the average derivatives over all input of the batch
        #as discussed in linear.py

        dLdA =  ( 2 * (self.A - self.Y) ) / (self.N * self.C)
        return dLdA




class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A=A #logits
        self.Y=Y
        self.N= A.shape[0]
        self.C= A.shape[1]
        #softmax activation
        self.softmax= Softmax().forward(A)  #NxC, each ith row corresponds to ith softmax output, which are predicted probabilities for ith input
        self.pred_probs_of_correct_out= np.sum((self.softmax * self.Y), axis=1) # this is a 1-d array (dim of N) of  probabilities for the correct label
        #divergenece wrt to each input is -log( predicted prob. for correct class)
        divergences= -np.log(self.pred_probs_of_correct_out)  #1-d array
        #loss for the batch is average of divergences
        self.L = np.sum(divergences)/self.N
        return self.L

    def backward(self):
        # here we shall plug in a simple formula that returns the derivative of
        #loss wrt the logits(input to softmax):
        # Loss= (1/N) * sum of divergences()
        # derivative of loss wrt a single logit: (1/N) * derivative of corresponding divergence wrt the logit
        # derivative of a divergernce wrt a logti is: softmax_output - ground_truth values
        # so we do this for each of the inputs and dLdA is a matrix of the shape NxC
        # whose ith row is the derivative of the loss wrt the ith logit vector
        #using dLdA we can now backprop easily as described in linear.py
        dLdA = (self.softmax-self.Y)/self.N  #the term N is a result of the loss itself being an average of divergences
        return dLdA


