#
# tsne_pytorch.py
# Created by Xingchang Huang on 9/9/2022
#

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim.optimizer import Optimizer, required


class MomentumSGD(Optimizer):
    """
    customized momentum-based gradient descent following: https://lvdmaaten.github.io/tsne/
    """

    def __init__(self, params, device=required, lr=required, n=required, no_dims=required):
        defaults = dict(lr=lr)
        super(MomentumSGD, self).__init__(params, defaults)

        self.device = device
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.eta = lr
        self.min_gain = 0.01
        self.n = n
        self.no_dims = no_dims
        self.dY = torch.zeros((self.n, self.no_dims)).float().to(device)
        self.iY = torch.zeros((self.n, self.no_dims)).float().to(device)
        self.gains = torch.ones((self.n, self.no_dims)).float().to(device)

    def __setstate__(self, state):
        super(MomentumSGD, self).__setstate__(state)

    def step(self, iter, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if iter < 20:
                    momentum = self.initial_momentum
                else:
                    momentum = self.final_momentum
                dY = d_p
                # print('dY:', dY.shape, dY)
                self.gains = (self.gains + 0.2) * ((dY > 0.) != (self.iY > 0.)) + (self.gains * 0.8) * ((dY > 0.) == (self.iY > 0.))
                self.gains[self.gains < self.min_gain] = self.min_gain
                # print('gains_torch:', iY.shape, gains.shape, dY.shape)
                self.iY = momentum * self.iY - self.eta * (self.gains * dY)
                p.data.add_(self.iY)
                p.data.add_(-torch.mean(p.data, 0).repeat(self.n, 1))

        return loss


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution. (copied from author's version)
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity (copied from author's version)
    """

    # Initialize some variables
    # print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        # if i % 500 == 0:
        #     print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    # print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def tsne_pytorch(X, init_Y, perplexity=20.0):
    eps = 1e-12
    no_dims = init_Y.shape[1]

    """ initialize the dissimilarity matrix with perplexity """
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.
    P = np.maximum(P, eps)
    # print('P:', P.shape)
    # P = P * 4.  # early exaggeration

    max_iter = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = X.shape[0]
    # Y = np.random.rand(n, no_dims)
    Y = init_Y

    """ original numpy Q matrix computation """
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0.
    Q1 = num / np.sum(num)

    """ pytorch Q matrix computation """
    Y = torch.from_numpy(Y).float().to(device)
    diagonal_ones = torch.eye(n).to(device)
    # Q = torch.cdist(Y.unsqueeze(0), Y.unsqueeze(0), p=2.0)[0] ** 2
    # Q = 1. / (1 + Q) - diagonal_ones
    # Q = Q / torch.sum(Q)
    # print('err:', np.mean((Q.detach().cpu().numpy() - Q1) ** 2))

    P = torch.from_numpy(P).float().to(device)
    # optimizer = torch.optim.Adam([Y.requires_grad_()], lr=10)
    optimizer = MomentumSGD([Y.requires_grad_()], device=device, lr=500, n=n, no_dims=no_dims)

    for iter in tqdm(range(max_iter)):
        optimizer.zero_grad()

        """ pytorch Q matrix computation """
        Q = torch.cdist(Y.unsqueeze(0), Y.unsqueeze(0), p=2.0)[0] ** 2
        Q = 1. / (1 + Q) - diagonal_ones
        Q = Q / torch.sum(Q)
        Q = torch.where(Q < eps, torch.ones_like(Q) * eps, Q)

        """ kl divergence """
        loss = torch.sum(P * torch.log(P / Q))
        loss.backward()

        """ autodiff """
        optimizer.step(iter)

        # if (iter + 1) % 10 == 0:
        #     print('iter: {:}, loss: {:.4f}'.format(iter, loss.item()))

        if iter == 100:
            P = P / 4.

    # print('Y:', Y.shape, Y.min(), Y.max())
    Y = Y.detach().cpu().numpy()
    # Y[:, 0] = (Y[:, 0] - Y[:, 0].min()) / (Y[:, 0].max() - Y[:, 0].min())
    # Y[:, 1] = (Y[:, 1] - Y[:, 1].min()) / (Y[:, 1].max() - Y[:, 1].min())
    return Y

#
# def main():
#     print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
#     print("Running example on 2,500 MNIST digits...")
#     X = np.loadtxt("mnist2500_X.txt")
#     labels = np.loadtxt("mnist2500_labels.txt")
#     # X = np.load('D:/xhuang/phd/repo/deepsampling/src/scripts/results_cluster/results_v4_latent_n1000/features_vgg_v19.npz')['features'].astype(np.float32)
#
#     Y = tsne_pytorch(X)
#
#     plt.figure(1)
#     plt.scatter(Y[:, 0], Y[:, 1], s=20, c=labels)
#     # plt.xlim([0, 1])
#     # plt.ylim([0, 1])
#     plt.gca().set_aspect('equal')
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
