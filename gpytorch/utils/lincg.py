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
    def __init__(self,max_iter=200,tolerance_resid=1e-5,precondition_closure=None):
        self.max_iter = max_iter
        self.tolerance_resid = tolerance_resid
        self.precondition_closure = precondition_closure

    def _diagonal_preconditioner(self,A,v):
        if v.ndimension() > 1:
            return v.mul((1/A.diag()).unsqueeze(1).expand_as(v))
        else:
            return v.mul((1/A.diag()).expand_as(v))

    def solve(self,A,b,x=None):
        if self.precondition_closure is None:
            self.precondition_closure = lambda v: self._diagonal_preconditioner(A,v)
            #self.precondition_closure = lambda v: self._silly_preconditioner(A,v)
#            self.precondition_closure = lambda v: v
#            pdb.set_trace()
            self._reset_precond = True

#        A.potrf()
        n = len(A)

#        if A.symeig()[0].min() < -1e-2:
#            pdb.set_trace()

        if b.ndimension() > 1:
            return self._solve_batch(A,b,x)

        if isinstance(A,Variable) or isinstance(b,Variable):
            raise RuntimeError('LinearCG is not intended to operate directly on Variables or be used with autograd.')

        if not isinstance(A,torch.Tensor) or not isinstance(b,torch.Tensor):
            raise RuntimeError('LinearCG is intended to operate on tensors.')

        if x is None:
            x = torch.zeros(n)

        residual = b - A.mv(x)
        z = self.precondition_closure(residual)
        p = z

        r_dot_z = residual.dot(z)

        for k in range(self.max_iter):
            Ap = A.mv(p)
            alpha = r_dot_z / p.dot(Ap)

            x = x + alpha*p
            residual = residual - alpha*Ap

            rtr = residual.dot(residual)
            if math.sqrt(rtr) < self.tolerance_resid:
                break

            z = self.precondition_closure(residual)

            new_r_dot_z = residual.dot(z)

            beta = new_r_dot_z / r_dot_z

            p = z + beta*p
            r_dot_z = new_r_dot_z

        if self._reset_precond:
            self.precondition_closure = None

        x_true = b.potrs(A.potrf())
#        if rtr > 1e-5:
            #pdb.set_trace()
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
        z = self.precondition_closure(residuals)
        P = z
        r_dot_zs = residuals.mul(z).sum(0)

        for k in range(min(self.max_iter,n)):
            AP = A.mm(P)
            PAPs = AP.mul(P).sum(0)

            alphas = r_dot_zs.div(PAPs)

            X = X + alphas.expand_as(P).mul(P)

            residuals = residuals - alphas.expand_as(AP).mul(AP)

            r_sq_news = residuals.pow(2).sum(0)

            if all(r_sq_news.sqrt().squeeze() < self.tolerance_resid):
                break

            z = self.precondition_closure(residuals)
            new_r_dot_zs = residuals.mul(z).sum(0)

            betas = new_r_dot_zs.div(r_dot_zs)

            P = z + betas.expand_as(P).mul(P)
            r_dot_zs = new_r_dot_zs

        if self._reset_precond:
            self.precondition_closure = None

        return X
