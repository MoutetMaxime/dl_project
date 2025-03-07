import numpy as np


class RBM:
    def __init__(self, p, q):
        # Xavier initialization
        self.w = np.random.randn(p, q) * np.sqrt(1 / p)
        self.a = np.zeros(p)  # Visible biases
        self.b = np.zeros(q)  # Hidden biases
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def entree_sortie(self, x):
        """ Compute P(h | v) """
        return self.sigmoid(x @ self.w + self.b)
    
    def sortie_entree(self, h):
        """ Compute P(v | h) """
        return self.sigmoid(h @ self.w.T + self.a)
    
    def train(self, x, epochs, batch_size, eps=0.01, verbose=True):
        N, p = x.shape
        q = self.b.shape[0]

        for epoch in range(epochs):
            x_copy = x.copy()
            np.random.shuffle(x_copy)

            for j in range(0, N, batch_size):
                x_batch = x_copy[j:min(j+batch_size, N)]
                t_batch = x_batch.shape[0]

                # Positive phase
                v_0 = x_batch
                p_h_v_0 = self.entree_sortie(v_0)
                h_0 = np.random.binomial(1, p_h_v_0)

                # Negative phase (reconstruction)
                p_v_h_0 = self.sortie_entree(h_0)
                v_1 = np.random.binomial(1, p_v_h_0)
                p_h_v_1 = self.entree_sortie(v_1)

                # Gradient updates
                grad_a = (v_0 - v_1).sum(axis=0)
                grad_b = (p_h_v_0 - p_h_v_1).sum(axis=0)
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1

                self.a += eps * grad_a / t_batch
                self.b += eps * grad_b / t_batch
                self.w += eps * grad_w / t_batch

            if verbose and epoch % 100 == 0:
                # Compute reconstruction error
                h = self.entree_sortie(x)
                x_rec = self.sortie_entree(h)
                error = np.mean((x - x_rec) ** 2)
                print(f"Epoch {epoch}: MSE = {error:.6f}")

    def generate(self, n, steps=100):
        """ Generate samples using Gibbs Sampling """
        v = np.random.rand(n, self.a.shape[0]) < 0.5  # Initialize random visible units

        for _ in range(steps):
            h = np.random.binomial(1, self.entree_sortie(v))  # Sample hidden states
            v = np.random.binomial(1, self.sortie_entree(h))  # Sample visible states

        return v  # Return generated samples


from load_data import lire_alpha_digit

characters = ['a', 'b', 'c', 'd', 'e']
images = lire_alpha_digit(characters)

rbm = RBM(320, 64)
rbm.train(images, epochs=10000, batch_size=64, eps=0.001, verbose=True)

# Generate new images
generated_images = rbm.generate(1)
print(generated_images.shape)
generated_images = generated_images.reshape(20, 16)

import matplotlib.pyplot as plt

plt.imshow(generated_images, cmap='gray')
plt.axis('off')
plt.show()
