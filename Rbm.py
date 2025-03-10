import numpy as np

from activations import sigmoid


class RBM:
    def __init__(self, p: int, q: int) -> None:
        self.w = np.random.randn(p, q) * np.sqrt(1 / p)
        self.a = np.zeros(p)  # Visible biases
        self.b = np.zeros(q)  # Hidden biases

    def entree_sortie(self, x: np.ndarray) -> np.ndarray:
        """ Compute P(h | v) """
        return sigmoid(x @ self.w + self.b)

    def sortie_entree(self, h: np.ndarray) -> np.ndarray:
        """ Compute P(v | h) """
        return sigmoid(h @ self.w.T + self.a)

    def train(self, x: np.ndarray, epochs: int, batch_size: int, eps: float=0.01, verbose: bool=True) -> None:
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
                h_0 = np.random.binomial(1, p_h_v_0)

                p_v_h_0 = self.sortie_entree(h_0)
                v_1 = np.random.binomial(1, p_v_h_0)
                p_h_v_1 = self.entree_sortie(v_1)

                grad_a = (v_0 - v_1).sum(axis=0)
                grad_b = (p_h_v_0 - p_h_v_1).sum(axis=0)
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1

                self.a += eps * grad_a / t_batch
                self.b += eps * grad_b / t_batch
                self.w += eps * grad_w / t_batch

            if verbose and epoch % 100 == 0:
                # Compute reconstruction error
                x_rec = self.sortie_entree(self.entree_sortie(x))
                error = np.mean((x - x_rec) ** 2)
                print(f"Epoch {epoch}: MSE = {error:.6f}")

    def generate(self, n: int, steps: int=100) -> np.ndarray:
        """ Generate samples using Gibbs Sampling """
        samples = np.zeros((n, self.a.shape[0]))
        p = self.a.shape[0]
        for i in range(n):
            x_new = (np.random.rand(p) < np.random.rand(p)).astype(int)

            for _ in range(steps):
                h = np.random.binomial(1, self.entree_sortie(x_new))
                x_new = np.random.binomial(1, self.sortie_entree(h))
            
            samples[i] = x_new
        return samples


if __name__ == "__main__":
    # Train the RBM on the binary alpha digit dataset
    from load_data import lire_alpha_digit

    characters = ["f", "m"]
    images = lire_alpha_digit(characters)
    print(images.shape)

    rbm = RBM(320, 64)
    rbm.train(images, epochs=30000, batch_size=16, eps=0.01, verbose=True)

    # Generate new images
    generated_images = rbm.generate(20)
    print(generated_images.shape)
    generated_images = generated_images.reshape(20, 20, 16)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(generated_images[i], cmap="gray")
        plt.axis("off")
    plt.show()