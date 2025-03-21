import numpy as np

from Rbm import RBM


class DBN:
    def __init__(self, sizes: list) -> None:
        self.rbms = [RBM(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    
    def train(self, x: np.ndarray, epochs: int, batch_size: int, rbm_steps: int=100, eps: float=0.001, verbose: bool=True) -> None:
        data = x.copy()
        for rbm in self.rbms:
            rbm.train(data, epochs, batch_size, eps, verbose=False)
            data = rbm.entree_sortie(data)
        if verbose:
            h = x.copy()
            for rbm in self.rbms:
                h = rbm.entree_sortie(h)
            for rbm in reversed(self.rbms):
                h = rbm.sortie_entree(h)
            error = np.mean((x - h) ** 2)
            print(f"MSE = {error:.6f}")

    def generate(self, n: int, steps: int=100) -> np.ndarray:
        samples = np.zeros((n, self.rbms[0].a.shape[0]))
        last_rbm_samples = self.rbms[-1].generate(n, steps)
        for i in range(n):
            sample = last_rbm_samples[i]
            for rbm in self.rbms[-2::-1]:
                sample = rbm.sortie_entree(sample)
                sample = np.random.binomial(1, sample)
            samples[i] = sample
        return samples

    def get_nb_params(self):
        return sum(rbm.get_nb_params() for rbm in self.rbms)