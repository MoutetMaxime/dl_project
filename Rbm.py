import numpy as np


class RBM:
    def __init__(self, p, q):
        self.w = np.random.randn(p, q) * np.sqrt(0.01)
        self.a = np.zeros(p)
        self.b = np.zeros(q)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def entree_sortie(self, x):
        return self.sigmoid(x @ self.w + self.b)
    
    def sortie_entree(self, h):
        return self.sigmoid(h @ self.w.T + self.a)
    
    def train(self, x, epochs, batch_size, eps=0.001, verbose=True):
        N, p = x.shape
        q = self.b.shape[0]

        for epoch in range(epochs):
            x_copy = x.copy()
            np.random.shuffle(x_copy)

            for j in range(0, N, batch_size):
                x_batch = x_copy[j:min(j+batch_size, N)]
                t_batch = x_batch.shape[0]

                v_0 = x_batch
                p_h_v_0 = self.entree_sortie(v_0)
                h_0 = (np.random.rand(t_batch, q) < p_h_v_0) * 1

                p_v_h_0 = self.sortie_entree(h_0)
                v_1 = (np.random.rand(t_batch, p) < p_v_h_0) * 1

                p_h_v_1 = self.entree_sortie(v_1)

                grad_a = (v_0 - v_1).sum(axis=0)
                grad_b = (p_h_v_0 - p_h_v_1).sum(axis=0)
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1

                self.a += eps * grad_a / t_batch
                self.b += eps * grad_b / t_batch
                self.w += eps * grad_w / t_batch
            
            h = self.entree_sortie(x)
            x_rec = self.sortie_entree(h)
            if verbose:
                print(((x - x_rec) ** 2 / (N * p)).sum())
    

rbm = RBM(100, 25)
data = np.random.rand(1000, 100)

rbm.train(data, 1000, 64, eps=0.01)
