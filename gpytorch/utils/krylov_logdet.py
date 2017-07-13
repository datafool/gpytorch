import torch
import pdb

class LanczosLogDet(object):
    def __init__(self,maxiter=float('inf'),num_random_probes=30):
        self.maxiter = maxiter
        self.num_random_probes = num_random_probes

    def lanczos(self,A,b):
        n = len(b)
        num_iters = min(self.maxiter,n)

        Q = torch.zeros(n,num_iters+1)
        alpha = torch.zeros(num_iters+1)
        beta = torch.zeros(num_iters)

        Q[:,0] = b/torch.norm(b)

        for j in range(num_iters):
            Q[:, j+1] = torch.mv(A,Q[:,j])
            alpha[j] = Q[:,j].dot(Q[:,j+1])
            Q[:,j+1] = Q[:,j+1]-alpha[j]*Q[:,j]
            if j > 0:
                Q[:,j+1] = Q[:,j+1] - beta[j-1]*Q[:,j-1]

            beta[j] = torch.norm(Q[:,j+1])
            Q[:,j+1] = Q[:,j+1]/beta[j]


        T = torch.diag(alpha) + torch.diag(beta,1) + torch.diag(beta,-1)
        T = T[:-1,:-1]
        Q = Q[:,:-1]
        return Q, T

    def logdet(self,A):
        n = len(A)
        V = torch.sign(torch.randn(n,self.num_random_probes))

        ld = 0

        for j in range(self.num_random_probes):
            vj = V[:,j]
            Q,T = self.lanczos(A,vj)

            # Eigendecomposition of a Tridiagonal matrix
            # O(n^2) time/convergence with QR iteration,
            # or O(n log n) with fast multipole?
            [f,Y] = T.eig(eigenvectors=True)
            f = f[:,0]

            ld = ld + n/(2.*self.num_random_probes) * (Y[:,0].pow(2).dot(f.log_()))

        return 2*ld