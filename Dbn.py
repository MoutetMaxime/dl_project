import numpy as np

from Rbm import RBM


class DBN:
    def __init__(self, sizes):
        self.rbms = [RBM(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    
    def train(self, x, epochs, batch_size, eps=0.001, verbose=True):
        for epoch in range(epochs):
            for rbm in self.rbms:
                rbm.train(x, epochs // len(self.rbms), batch_size, eps, verbose=False)
                x = rbm.entree_sortie(x)

            if verbose and epoch % 10 == 0:
                # Compute reconstruction error
                h = x.copy()
                for rbm in self.rbms[::-1]:
                    h = rbm.sortie_entree(h)
                error = np.mean((x - h) ** 2)
                print(f"Epoch {epoch}: MSE = {error:.6f}")

    def generate(self, n):
        x = np.random.rand(n, self.rbms[0].a.shape[0]) < 0.5
        for rbm in self.rbms:
            x = rbm.entree_sortie(x)
        return x

# Train the DBN on the binary alpha digit dataset

from load_data import lire_alpha_digit

characters = ['a', 'b', 'c', 'd', 'e']
images = lire_alpha_digit(characters)

dbn = DBN([320, 64, 64, images.shape[-1]])
dbn.train(images, epochs=100, batch_size=64, eps=0.0001)

# Generate new images
generated_images = dbn.generate(1)
print(generated_images.shape)
generated_images = generated_images.reshape(20, 16)

import matplotlib.pyplot as plt

plt.imshow(generated_images, cmap='gray')
plt.axis('off')
plt.show()
