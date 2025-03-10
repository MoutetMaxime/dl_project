import numpy as np

from Dbn import DBN
from Rbm import RBM


class DNN(DBN):
    def __init__(self, sizes: list) -> None:
        super(DNN).__init__(sizes)
        self.cls_layer = self.rbms[-1]
        self.rbms = self.rbms[:-1]

    def pretrain(self, x: np.ndarray, epochs: int, batch_size: int, eps: float=0.001, verbose: bool=True) -> None:
        super(DNN).train(x, epochs, batch_size, eps, verbose)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - x.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def calcul_softmax(rbm: RBM, x: np.ndarray) -> np.ndarray:
        return DNN.softmax(x @ rbm.w + rbm.b)
