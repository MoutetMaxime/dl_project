import numpy as np

from activations import softmax
from Dbn import DBN
from Rbm import RBM


class DNN(DBN):
    def __init__(self, sizes: list) -> None:
        super(DNN, self).__init__(sizes)
        self.cls_layer = self.rbms[-1]
        self.rbms = self.rbms[:-1]

    def pretrain(self, x: np.ndarray, epochs: int, batch_size: int, eps: float=0.001, verbose: bool=True) -> None:
        super(DNN, self).train(x, epochs, batch_size, eps, verbose)

    @staticmethod
    def calcul_softmax(rbm: RBM, x: np.ndarray) -> np.ndarray:
        return softmax(x @ rbm.w + rbm.b)

    def entree_sortie_reseau(self, x: np.ndarray):
        activations = []
        current = x
        for rbm in self.rbms:
            current = rbm.entree_sortie(current)
            activations.append(current)
        out = self.calcul_softmax(self.cls_layer, current)
        return activations, out

    def retropropagation(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int, verbose: bool=True):
        n = x.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = x[batch_idx]
                y_batch = y[batch_idx]
                activations, out = self.entree_sortie_reseau(x_batch)
                grad = out - y_batch
                gb = grad.sum(axis=0)
                gw = activations[-1].T @ grad
                self.cls_layer.b -= lr * gb / batch_size
                self.cls_layer.w -= lr * gw / batch_size
            if verbose:
                _, out_all = self.entree_sortie_reseau(x)
                loss = -np.mean(np.sum(y * np.log(out_all + 1e-9), axis=1))
                print(epoch, loss)

    def test_DNN(self, x_test: np.ndarray, y_test: np.ndarray):
        _, out = self.entree_sortie_reseau(x_test)
        preds = np.argmax(out, axis=1)
        true = np.argmax(y_test, axis=1)
        return np.mean(preds != true)
