import numpy as np
import sys

sys.path.append("mytorch")
from mytorch.nn.linear import Linear
from mytorch.rnn_cell import RNNCell


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class. as given in the handout"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn = [
            (
                RNNCell(input_size, hidden_size)
                if i == 0
                else RNNCell(hidden_size, hidden_size)
            )
            for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size)

        Output (y): logits

        """
        #we have made the rnn so, such that all we do is define the structure of a
        #single MLP used in the rnn. Notice that in the __init__ method we make no assumptions
        #about the no. of time steps. We just define the no. of hidden layer (rnn cells)
        #rest is dependent on the input. Essentially, all we have is a single MLP,
        # it is left to the input dimensions to decide how many times will it be used recursively
        #this makes our approach so modular
        self.x=x
        batches, self.seqLen , inSize= x.shape
        if h_0 is None:
            h_0 = np.zeros(shape=(self.num_layers, batches, self.hidden_size))


        self.hiddens.append(h_0.copy())  # to store hidden state outputs at each timestep for each layer
        hiddenStates=h_0.copy()  #to store hidden states of the previous timestep

        for time_step in range(self.seqLen):
            for layer in range(self.num_layers):
                if layer==0: #for the  first layer, we use x as inputs
                    #hiddenStates[0] stores the output of the hidden state, which will be passed as input to next layer
                    hiddenStates[layer] = self.rnn[layer].forward(x[:, time_step, :], hiddenStates[layer])
                else:
                    #input to cell is previous layers' output(layer-1) and the hidden state of the previous time step (layer)
                    hiddenStates[layer] = self.rnn[layer].forward(hiddenStates[layer-1], hiddenStates[layer])

            #now hiddenStates stores the hidden states of the current_time, these will be needed
            #in the backward pass, so we store these
            self.hiddens.append(hiddenStates.copy()) #not using .copy gave me so much headache for hours

        print(f"hiddens.shape: {len(self.hiddens)}, {self.hiddens[0].shape}")
        logits = self.output_layer.forward(self.hiddens[self.seqLen][-1])
        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, output_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """

        #derivative wrt the final hidden state of the last time step
        dh_prev_l = self.output_layer.backward(delta) # batch_size x hidden size
        batch_size, hidden_size = dh_prev_l.shape

        hidden_grads = np.zeros(shape= (self.seqLen+1, self.num_layers, batch_size, hidden_size))
        hidden_grads[self.seqLen, self.num_layers-1] = dh_prev_l


        for time_step in range(self.seqLen, 0, -1):

            print(f"time step: {time_step}")
            for layer in range(self.num_layers-1, -1, -1) :
                if (layer==0):
                    #for the very first rnn cell, the input (h_prev_l) is x, which is the actual input
                    dx, dh_prev_t= self.rnn[layer].backward(hidden_grads[time_step, layer],
                                                                       self.hiddens[time_step][layer],
                                                                       self.x[:, time_step-1, :],
                                                                       self.hiddens[time_step-1][layer]
                                                                       )
                    hidden_grads[time_step-1, layer] += dh_prev_t

                else:

                    dh_prev_l, dh_prev_t = self.rnn[layer].backward(hidden_grads[time_step, layer],
                                                                          self.hiddens[time_step][layer],
                                                                          self.hiddens[time_step][layer-1], # 5 x 32 but its 32 x 40
                                                                          self.hiddens[time_step-1][layer]
                                                                          )
                    hidden_grads[time_step, layer-1] += dh_prev_l
                    hidden_grads[time_step-1, layer] += dh_prev_t

        return hidden_grads[0]/ batch_size