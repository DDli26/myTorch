import numpy as np


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
        square_error= np.square(self.A - self.Y)
        self.L= np.mean(square_error, axis=1)  #MSE

        return self.L


    def backward(self):
        pass


# a= np.random.randint(1,5, size=(2,2))
# y=np.random.randint(1,5, size=(2,2))
# print("a:\n",a,"\n\nY:\n", y)
# print("\nsquare_error\n")
# print(np.square(a - y))
# print("\nMSE\n")
# print( np.mean(np.square(a - y), axis=1).reshape())

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
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        self.softmax = np.exp(A) / np.sum(np.exp(A), axis=1).reshape(-1, 1)

        crossentropy = (-Y * np.log(self.softmax)) @ Ones_C
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.N

        return dLdA
