# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/24/19
# Time: 1:16 PM
# File: objective.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def dcca_objective(pred1, pred2, reg):
    """
    Implements the DCCA objective (correlation maximization. It assumes pred1 and pred2 are in row major format
    i.e. [n_samples, n_output_nodes]
    So each view's data is first transposed before being used for calculating the centered data matrix.

    :param pred1: Predictions from view 1 model
    :param pred2: Predictions from view 2 model
    :param reg: regularization parameter. Must be > 0
    :return: corr, dH1, dH2
    """

    H1 = torch.Tensor(pred1.data).t()
    H2 = torch.Tensor(pred2.data).t()

    o = H1.shape[0]  # latent dimension
    m = H1.shape[1]  # number of training samples
    ones = torch.ones(m, m)
    I = torch.eye(o, o)  # identity matrix

    # centered data matrices
    H1_hat = H1.sub(torch.mul(H1.mm(ones), (1.0 / m)))
    H2_hat = H2.sub(torch.mul(H2.mm(ones), (1.0 / m)))

    SigmaHat12 = torch.mul(H1_hat.mm(torch.t(H2_hat)), 1.0 / (m - 1))
    SigmaHat11 = torch.add(torch.mul(H1_hat.mm(torch.t(H1_hat)), 1.0 / (m - 1)), torch.mul(reg, I))
    SigmaHat22 = torch.add(torch.mul(H2_hat.mm(torch.t(H2_hat)), 1.0 / (m - 1)), torch.mul(reg, I))

    # SVD decomposition for square root calculation
    D1, V1 = torch.eig(SigmaHat11, eigenvectors=True)
    D1 = D1[:, 0]
    epsilon = 1e-12
    D1_indices = torch.nonzero(D1.squeeze() > epsilon)
    D1 = D1[D1_indices].squeeze()
    V1 = V1[D1_indices].squeeze()

    D2, V2 = torch.eig(SigmaHat22, eigenvectors=True)
    D2 = D2[:, 0]
    D2_indices = torch.nonzero(D2.squeeze() > epsilon)
    D2 = D2[D2_indices].squeeze()
    V2 = V2[D2_indices].squeeze()

    # calculate root inverse of correlation matrices
    SigmaHat11RootInv = torch.inverse(V1.mm(torch.pow(torch.diag(D1), 0.5)).mm(torch.t(V1)))
    SigmaHat22RootInv = torch.inverse(V2.mm(torch.pow(torch.diag(D2), 0.5)).mm(torch.t(V2)))

    # Total correlation
    T = SigmaHat11RootInv.mm(SigmaHat12).mm(SigmaHat22RootInv)
    corr = torch.sqrt(torch.trace(torch.t(T).mm(T)))
    # corr = torch.trace(torch.sqrt(torch.t(T).mm(T)))

    # Gradient calculations
    U, D, V = torch.svd(T)

    Delta12 = SigmaHat11RootInv.mm(U).mm(torch.t(V)).mm(SigmaHat22RootInv)
    Delta11 = torch.mul(SigmaHat11RootInv.mm(U).mm(torch.diag(D)).mm(U.t()).mm(SigmaHat11RootInv), -0.5)
    Delta22 = torch.mul(SigmaHat22RootInv.mm(U).mm(torch.diag(D)).mm(U.t()).mm(SigmaHat22RootInv), -0.5)

    # dcorr(H1, H2) / dH1 and dcorr(H1, H2) / dH2
    dH1 = torch.mul(torch.mul(Delta11.mm(H1_hat), 2.0) + Delta12.mm(H2_hat), 1.0 / (m - 1))
    dH2 = torch.mul(torch.mul(Delta22.mm(H2_hat), 2.0) + Delta12.mm(H1_hat), 1.0 / (m - 1))

    return corr, dH1.t(), dH2.t(), SigmaHat11RootInv.mm(U), SigmaHat22RootInv.mm(V)
