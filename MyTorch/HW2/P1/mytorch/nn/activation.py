import numpy as np
import scipy

#remember, activation function are applied element-wise
# N is batch size
# Cout is the no. of output neurons in the layer
# dimension(Z) = NxCout
# dimension(A)= NxCout
# dimension(dLdA)= NxCout
# dimension(dLdZ) = NxCout
# dLdZ is calculated by element wise multiplication of dLdA * dAdZ
# dAdZ is also calculated by elementwise operation on the activations A


class Identity:
    """
    Identity activation is no activation, so we really we have to just
    return the input to the forward and backwork functions
    """
    def forward(self, Z):

        #Z is NxCout
        #A is NxCout and is the activation
        self.A= Z
        return self.A

    def backward(self, dLdA):
        #dLdA is NxCout
        self.dLdZ=dLdA
        return self.dLdZ



class Sigmoid:

    def forward(self, Z):
        """
         perform element wise sigmoid
        """
        self.A= np.ones_like(Z)/(1+ np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        """
            backprop through sigmoid activation
            peforms an element wise derivative
            dLdA is (NxCout)
            dLdZ will also be (NxCout)
        """
        dAdZ= self.A*(1-self.A)
        dLdZ= dLdA * dAdZ
        return dLdZ


class Tanh:

    def forward(self, Z):
        """
        again, element wise tanh
        """
        self.A = np.exp(Z)-np.exp(-Z)
        self.A= self.A/ (np.exp(Z) + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ= 1 - np.square(self.A)
        dLdZ= dLdA * dAdZ
        return dLdZ


class ReLU:

    def forward(self, Z):
        self.A= np.maximum(0, Z) #negative elements are replaced by 0
        return self.A

    def backward(self, dLdA):
        dAdZ= np.where(self.A>0, 1, 0)
        dLdZ= dLdA * dAdZ
        return dLdZ


class GELU:
    """
    Read the writeup for further details on GELU.
    """

    def forward(self, Z):
        self.Z=Z
        self.A = 0.5*Z*( 1 + scipy.special.erf(Z/np.sqrt(2)))
        return self.A
    def backward(self, dLdA):
        dAdZ= 0.5*(1+scipy.special.erf(self.Z/np.sqrt(2))) + (
            (self.Z/np.sqrt(2*np.pi)) * (np.exp(-np.square(self.Z)/2))
        )

        self.dLdZ= dLdA * dAdZ
        return self.dLdZ



class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        #Z and output A will both be NxCout
        # ith row corresponds to ith input of the batch
        # in softmax every element of a row in A depends on every element of the
        # corresponding row of Z

        exp=np.exp(Z)
        #find the row-wise sum of exp
        exp_sum = np.sum(exp, axis=1)
        exp_sum = np.reshape(exp_sum, (-1, 1))
        self.A =exp/exp_sum
        return self.A


    def backward(self, dLdA):

        #firstly, for a given activation vector A and a preactivation vector Z
        #dAdZ is a Jacobian J of the shape (Cout x Cout)
        #dLdZ is the gradient vector and is given by dLdA @ dAdZ
        #you see for each A and Z pair, there is a different jacobian and hence, this operation cannot be
        # so easily handled by vectorization
        #we'll have to do it for each A and Z pair separately
        #total N such pairs

        # Calculate the batch size and number of features
        batch_size=dLdA.shape[0]
        num_features=dLdA.shape[1]

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ=np.zeros_like(dLdA)


        # Fill dLdZ one data point (row) at a time
        for row in range (batch_size):

            # Initialize the Jacobian with all zeros.
            jacobian= np.zeros(shape=(num_features, num_features))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for i in range(num_features):
                for j in range(num_features):
                    if i==j:
                        jacobian[i][j]= self.A[row,i] * (1- self.A[row, i])
                    else:
                        jacobian[i][j]= -(self.A[row, i]) * self.A[row, j]


            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[row] = dLdA[row] @ jacobian

        return dLdZ

