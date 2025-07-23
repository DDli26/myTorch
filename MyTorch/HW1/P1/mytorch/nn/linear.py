import numpy as np

class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.

         W.T is of shape Cin x Cout, ith column contains weights for the ith neuron of the next layer
        jth row of W.T contains the weights for the jth feature of input (each input is of size C0)
        we like to use the shape CoutxCin for W and not the more natural CinxCout because CoutxCin fits in the formulae
        more easily. as you'll see
        """

        self.W=np.zeros(shape=(out_features, in_features))
        self.b=np.zeros(shape=(out_features, 1))


    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details

        N is the batch size
        Cin is the no. of features per input
        Cout is therefore the no of output neurons
        A is of shape (N, Cin), ith row has the ith input's features
        W.T is of shape Cout x Cin, ith col contains weights for the ith neuron of the next layer
        jth row of W contains the weights for the jth feature of input (each input is of size C0)

        to get forward pass for a single input of the shape 1xC0 we do,
        input(1xCin) @ W.T(CinxCout) ,resulting in a output for next layer of shape 1xC1
        ith element of 1xC1 is the preactivation z for the ith neuron of the output, but without the bias

        to get output for the entire batch (NxC0) we do
        A(NxC0) @ W.T(C0xC1)
        resulting in a batch of output NxC1,
        where ith row is the output for ith input of the batch
        """


        self.A=A
        self.N= A.shape[0]
        Z= self.A @ self.W.T   # this would be (N,C0) multiplied by (C0, C1) --> NxC1
        Z=Z+self.b.T  #b.T is of shape (1,C1) and thus it will be broadcasted to NxC1

        return Z

    def backward(self, dLdZ):
        """
        in this function , we shall compute the gradient wrt the parameters of this layer object,
        we must ensure that the derivative matrices or vectors match the dimension of the matrices or vectors
        we are finding the derivative for, this would make gradient descent easier for us

        we also want to  compute the derivative wrt to the input of the layer: A(NxCout),
        so that it can be propagated back to previous layers
        Cin--> no. of input neurons for this layer object
        Cout--> no. of output neurons for this layer object

        dLdZ is the derivative wrt the affine sums of the next layer, shape: NxCout

        dLdA is the derivative wrt the batch of input
        1 input is of the form 1xCin
        A is of the form NxCin
        dLdA will also be of the form NxCin, where N is the batch size
        now for the case of a batch of 1,
        dLdA= 1xCin, remember, the gradient vector is a row vector
        dLdZ=1xCout, again gradient is a row vector
        Z= A @ W.T +b.T
        dLdA if you hand calculate (that's what i did to verify), comes out to be dLdZ(1xCout). W(CoutxCin)
        that is a row vector: dLdA of the form 1xCin

        dLdA(NxCin) is dlDZ(NxCout).W(Cout x Cin)
        ith row of dLdA contains derivatives for the ith input
        """
        self.dLdA= dLdZ @ self.W

        #coming to dLdW and dLdb, now we know that the derivative of loss(scalar) with respect to a given parameter
        #w(l,i,j) or b(l,k) is a sum of the derivatives of w(l, i, j) with respect to all the individual input
        #divergences divided by the total no. of inputs.
        #you'll see that vectorization (A of the form (NxCin) and dLdZ_transpose(CoutxN)) makes the calculation
        #really easy
        #to find dLdW again if you hand calculate you'll see that dLdZ_transpose(CoutxN) @ A(NxCin)
        # gives dLdW(CoutxCin), in this matrix, the ij entry is the average of the derivatives of the ij weight
        # with respect to each individual input. and, really, with just one matrix multiplication we are done.
        # In this the ij entry of dLdW contains the sum of derivatives (verify) of wij with respect to all the
        # individual divergences and as for the average, the no. by which we divide (mini-batch size) is carried over
        # as a result  of backpropagating the loss
        # you see, dLdZ/dLdA is of the form NxCout and divergence wrt a given z(l, i) differs for every input
        # and that is why we keep a separate row for each input
        # but look at how convenient it is for calculating dLdW and dLdb
        self.dLdW= dLdZ.T @ self.A

        #now just like dLdW
        #dLdb will also be a single vector, where ith entry is the average
        # bias wrt the ith input of the mini-batch

        self.dLdb= np.sum(dLdZ, axis=0).reshape(-1, 1) # Coutx1, to match the dimensions of self.b

        return self.dLdA




# testing that the forward pass implementation works
# np.random.seed(42)
#
# W=np.random.randint(1,5, size=(3,2)) #3 outputs and 2 inputs
# print("W.T\n",W.T)
# b=np.random.randint(1,5, size=(3,1))
# print(f"\nb\n {b}")
# print(f"shape of b is {b.shape} shape of W.T is {(W.T).shape}")
# A=np.random.randint(1,3, size=(4,2)) #batch of 4, each input of size 2
# print(f"\nA:\n {A} \nshape of A is {A.shape} ")
# A_into_W=A@W.T
# print(f"\nA@W is \n {A_into_W} \n shape is {A_into_W.shape}")
#
# print(f"\n\nAdding the bias: \n{A_into_W+b.T}")

