import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU

# Z0 means the preactivations of the first layer
#A1 is the activation of the first layer
# i know Z1 and A1 would be a better name pair, but really those are the names the autograder
#uses for evaluation so we can't do anything
#similarly Zi means the preactivations of the i+1 layer,
# and A(i+1) means the activations of the i+1 layer
# also the dimensions of the layers are really according to the diagrams of the
# network, as shown in the writeup
#note that here we are just concerned with finding the ouput of the last layer,
#we are not calculating loss for now

class MLP0:
    """
    MLP0 is a Multi Layer Perceptron with 0 hidden layers
    inside __init__ we will have a variable named layers which is a list
    of all the layers/activations in sequence
    these layers/activations are object from nn.linear and nn.activation
    forward and backward methods simply use layers list to perform
    forward pass and backward pass
    Z0 stores the preactivations for the layer following the input layer
    A1 stores the activations for Z0
    """
    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """
        self.layers=[Linear(2,3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layer followed by the activation layer to get the model output.
        """
        #A0 is the input, this would be of dimension Nx2 where N is the batch size

        Z0=self.layers[0].forward(A0)
        A1= self.layers[1].forward(Z0)

        if self.debug:
            self.Z0=Z0
            self.A1=A1
        return A1

    def backward(self, dLdA1):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        once the backward pass is done, all layer
        objects will have the derivaties that we need for gradient descent stored inside them
        """
        #dLdA1 stores the derivatives of the loss wrt final ouput of the final layer of self.layers
        #or dLdA1 is simply the derivative wrt the output of the activation layer, self.layers[1]
        #we have propagate the error back to calculate the gradients wrt to all the parameters
        #this will be done once we call backward() for each of the layers, starting with the last layer
        #dLdZ0 is the derivative wrt the affine sums of the first layer (there is just the input layer and the
        #dLdA0 is the derivative wrt the input
        dLdZ0= self.layers[1].backward(dLdA1)
        dLdA0=self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ0=dLdZ0
            self.dLdA0= dLdA0
        return dLdA0


class MLP1:

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """

        self.layers=[Linear(2,3), ReLU(), Linear(3, 2), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        #this is just like we did it for MLP0, if you understand MLP0.forward, this makes sense
        Z0=self.layers[0].forward(A0)
        A1=self.layers[1].forward(Z0)
        Z1=self.layers[2].forward(A1)
        A2=self.layers[3].forward(Z1)

        if self.debug: #for the autograder
            self.Z0=Z0
            self.A1=A1
            self.Z1=Z1
            self.A2=A2
        return A2

    def backward(self, dLdA2):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """
        #again, this is same as MLP0, just with more layers, notations are as in the writeup or
        #as commented at the start of the file
        #dLdA2 is the derivative wrt the output of the last actiation
        dLdZ1= self.layers[3].backward(dLdA2)
        dLdA1=self.layers[2].backward(dLdZ1)
        dLdZ0=self.layers[1].backward(dLdA1)
        dLdA0=self.layers[0].backward(dLdZ0)

        if self.debug: #for the autograder

            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagramatic view in the writeup for better understanding.
        Use ReLU activation function for all the linear layers.)
        """

        # List of Hidden and activation Layers in the correct order
        self.layers=[Linear(2,4),
                     ReLU(),
                     Linear(4,8),
                     ReLU(),
                     Linear(8, 8),
                     ReLU(),
                     Linear(8,4),
                     ReLU(),
                     Linear(4,2),
                     ReLU()
                     ]

        self.debug = debug

    def forward(self, A):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        if self.debug:
            self.A = [A]

        #we loop through the layers this time
        for layer in self.layers:
            A=layer.forward(A) #A stores the output of the current layer
            if self.debug:
                self.A.append(A)

        return A

    def backward(self, dLdA):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        if self.debug:
            self.dLdA = [dLdA]

        for i in reversed(range(len(self.layers))):
            dLdA=  self.layers[i].backward(dLdA)

            if self.debug: #for the debugger
                self.dLdA= [dLdA] + self.dLdA

        return dLdA
