import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.h_prev_t = h_prev_t
        self.x = x
        self.hidden = h_prev_t
        self.r_activ = Sigmoid()
        self.r = self.r_activ.forward(
                 self.Wrx @ x + self.brx +
                 self.Wrh @ h_prev_t + self.brh
        )
        self.z_activ = Sigmoid()
        self.z = self.z_activ.forward(
                self.Wzx @ x + self.bzx +
                self.Wzh @ h_prev_t + self.bzh
        )
        self.n_activ = Tanh()
        self.n = self.n_activ.forward(
                self.Wnx @ x + self.bnx +
                self.r * (self.Wnh @ h_prev_t + self.bnh)
        )

        h_t =  (1 - self.z) * self.n + self.z * h_prev_t
        return h_t


    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        #eq1 from handout
        #a
        dh_dz = self.h_prev_t - self.n
        dldz = delta * (dh_dz)

        #b
        dldn = delta * (1 - self.z)

        #eq2 from handout
        #a
        dl_dn_preactiv = self.n_activ.backward(dldn) # dimension: hout
        self.dWnx = dl_dn_preactiv.reshape(-1, 1) @ self.x.reshape(1, -1)

        #b
        self.dbnx = dl_dn_preactiv

        #c: in case of error, double check this
        dl_drt = dl_dn_preactiv * (self.Wnh @ self.h_prev_t + self.bnh)

        #d
        self.dWnh = dl_dn_preactiv.reshape(-1, 1) * np.outer(self.r, self.h_prev_t)
        #e
        self.dbnh = dl_dn_preactiv * self.r

        # eq3
        #a : similar to the code for self.dWnx
        dl_dz_preactiv = self.z_activ.backward(dldz)  # dimension: hout
        self.dWzx = np.outer(dl_dz_preactiv, self.x)

        #b
        self.dbzx = dl_dz_preactiv

        #c
        self.dWzh = np.outer(dl_dz_preactiv, self.h_prev_t)

        #d
        self.dbzh = dl_dz_preactiv

        #e4: this is the same as eq3 but with change of variables
        #a
        dl_dr_preactiv = self.r_activ.backward(dl_drt)
        self.dWrx = np.outer(dl_dr_preactiv, self.x)

        #b
        self.dbrx = dl_dr_preactiv

        #c
        self.dWrh = np.outer(dl_dr_preactiv, self.h_prev_t)

        #d
        self.dbrh = dl_dr_preactiv

        #eq 5
        dx = ( dl_dn_preactiv @ self.Wnx )  + ( dl_dz_preactiv @ self.Wzx ) + (dl_dr_preactiv @ self.Wrx)

        #dl_dh_prev_t: we'll have to go down 4 paths
        dh = ((delta * self.z) +
              (dl_dn_preactiv @ ( self.Wnh * self.r.reshape(-1, 1) ) ) +
              (dl_dz_preactiv @ self.Wzh) +
              (dl_dr_preactiv @ self.Wrh))

        return dx, dh




