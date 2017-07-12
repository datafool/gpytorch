import math
import torch
from torch.autograd import Variable

import pdb

class LinearCG(object):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        Ax = b

    for positive definite and symmetric matrices A.
    """
    def __init__(self,max_iter=3,tolerance_resid=1e-5):
        self.max_iter = max_iter
        self.tolerance_resid = tolerance_resid

    def solve(self,A,b,x=None):
        if isinstance(A,Variable) or isinstance(b,Variable):
            raise RuntimeError('LinearCG is not intended to operate directly on Variables or be used with autograd.')

        if not isinstance(A,torch.Tensor) or not isinstance(b,torch.Tensor):
            raise RuntimeError('LinearCG is intended to operate on tensors.')

        n = len(b)

        if x is None:
            x = torch.randn(n)

        residual = b - A.mv(x)
        p = residual
        r_sq_old = residual.dot(residual)

        for k in range(min(self.max_iter,n)):
            Ap = A.mv(p)
            alpha = r_sq_old / p.dot(Ap)

            #x = x + alpha*p
            x.add_(alpha*p)

            residual = residual - alpha*Ap

            r_sq_new = residual.dot(residual)

            if math.sqrt(r_sq_new) < self.tolerance_resid:
                break

            beta = r_sq_new / r_sq_old

            p = residual + beta*p
            #p.mul_(beta).add_(residual)
            r_sq_old = r_sq_new

        return x

    def _solve_batch(self,A,B,X=None):
        n, k = B.size()
        if isinstance(A,Variable) or isinstance(B,Variable):
            raise RuntimeError('LinearCG is not intended to operate directly on Variables or be used with autograd.')

        if not isinstance(A,torch.Tensor) or not isinstance(B,torch.Tensor):
            raise RuntimeError('LinearCG is intended to operate on tensors.')

        if X is None:
            X = torch.randn(n,k)

        residuals = B - A.mm(X)
        P = residuals

        r_sq_olds = residuals.pow(2).sum(0)

        for k in range(min(self.max_iter,n)):
            AP = A.mm(P)
            PAPs = AP.mul(P).sum(0)

            alphas = r_sq_olds.div(PAPs)

            X = X + alphas.expand_as(P).mul(P)

            residuals = residuals - alphas.expand_as(AP).mul(AP)

            r_sq_news = residuals.pow(2).sum(0)

            if all(r_sq_news.sqrt().squeeze() < self.tolerance_resid):
                break

            betas = r_sq_news.div(r_sq_olds)

            P = residuals + betas.expand_as(P).mul(P)
            r_sq_olds = r_sq_news

        return X
