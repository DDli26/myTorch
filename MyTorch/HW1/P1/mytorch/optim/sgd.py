import numpy as np


class SGD:
    """
    Class attributes:
        – l: list of model layers
        – L: number of model layers
        – lr: learning rate, tunable hyperparameter scaling the size of an update.
        – mu: momentum rate µ, tunable hyperparameter controlling how much the previous updates affect
        the direction of current update. µ = 0 means no momentum.
        – v W: list of weight velocity for each layer
        – v b: list of bias velocity for each layer

    Class methods:
        – step: Updates W and b of each of the model layers:
        ∗ Because parameter gradients tell us which direction makes the model worse, we move opposite
        the direction of the gradient to update parameters.
        ∗ When momentum is non-zero, update velocities v_W and v_b, which are changes in the gradient
        to get to the global minima. The velocity of the previous update is scaled by hyperparameter
        µ, refer to lecture slides for more details.

    """
    def __init__(self, model, lr=0.1, momentum=0):
        self.l= model.layers #its a list of all the individual layer objects of the model
        print("Printing model layers, just to see what they look like:\n ", model.layers)
        self.L= len(self.l)
        self.lr=lr
        self.mu=momentum
        self.v_W= [np.zeros_like(self.l[i].W) for i in range(self.L)] #wouldn't this fail? because the activation layers have no parameters?
        self.v_b=[np.zeros_like(self.l[i].b) for i in range(self.L)]


    def step(self):
        
        #this code does not account for the fact that some layers don't have any parameters
       if self.mu==0: #no momentum

            #update parameters for each layer
            for i  in range(self.L):
                self.l[i].W= self.l[i].W - self.lr * (self.l[i].dLdW)
                self.l[i].b= self.l[i].b - self.lr * (self.l[i].dLdb)

       else:
            for i in range(self.L):
                #first we must find the gradient with momentum
                self.v_W[i]= self.l[i].dLdW + self.mu * (self.v_W[i])
                self.v_b[i]= self.l[i].dLdb + self.mu * (self.v_b[i])

                #now we perform the updates
                self.l[i].W = self.l[i].W - self.lr * (self.v_W[i])
                self.l[i].b = self.l[i].b - self.lr * (self.v_b[i])

