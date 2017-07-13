import torch
import pdb
import math

class LanczosLogDet(object):
    def __init__(self,maxiter=15,num_random_probes=500):
        self.maxiter = maxiter
        self.num_random_probes = num_random_probes

    def lanczos(self,A,b):
        n = len(b)
        num_iters = min(self.maxiter,n)

        Q = torch.zeros(n,num_iters+1)
        alpha = torch.zeros(num_iters)
        beta = torch.zeros(num_iters)

        b = b/torch.norm(b)
        u = torch.zeros(n)

        Q[:,0] = b

        for k in range(1,num_iters+1):
            u,b,alpha_k,beta_k = self._lanczos_step(u,b,A,Q[:,:k])
            alpha[k-1] = alpha_k
            beta[k-1] = beta_k
            Q[:,k] = u

            if math.fabs(beta[k-1])<1e-5:
                break

        beta = beta[1:]
        T = torch.diag(alpha) + torch.diag(beta,1) + torch.diag(beta,-1)
        return Q[:,1:], T


    def _lanczos_step(self,u,v,B,Q):
        norm_v = torch.norm(v)
        orig_u = u

        u = v/norm_v

        if Q.size()[1] == 1:
            u = u - Q.mul((Q.t().mv(u)).expand_as(Q))
        else:
            u = u - Q.mv(Q.t().mv(u))

        u = u/torch.norm(u)

        r = B.mv(u) - norm_v*orig_u

        a = u.dot(r)
        v = r - a*u
        pdb.set_trace()

        return u,v,a,norm_v

    def logdet(self,A):
        n = len(A)
        V = torch.randn(n,self.num_random_probes)

        ld = 0
        pdb.set_trace()

        for j in range(self.num_random_probes):
            vj = V[:,j]
            Q,T = self.lanczos(A,vj)

            # Eigendecomposition of a Tridiagonal matrix
            # O(n^2) time/convergence with QR iteration,
            # or O(n log n) with fast multipole?
            [f,Y] = T.symeig(eigenvectors=True)

            ld = ld + n/(2.*self.num_random_probes) * (Y[:,0].pow(2).dot(f.log_()))

        return 2*ld