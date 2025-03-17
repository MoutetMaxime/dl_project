import numpy as np

from Rbm import RBM


# class DBN:
#     def __init__(self, sizes: list) -> None:
#         self.rbms = [RBM(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    
#     def train(self, x: np.ndarray, epochs: int, batch_size: int, rbm_steps: int=100, eps: float=0.001, verbose: bool=True) -> None:
#         for epoch in range(epochs):
#             for rbm in self.rbms:
#                 rbm.train(x, rbm_steps, batch_size, eps, verbose=False)
#                 x = rbm.entree_sortie(x)

#             if verbose and epoch % 10 == 0:
#                 # Compute reconstruction error
#                 h = x.copy()
#                 for rbm in self.rbms[::-1]:
#                     h = rbm.sortie_entree(h)
#                 error = np.mean((x - h) ** 2)
#                 print(f"Epoch {epoch}: MSE = {error:.6f}")

#     def generate(self, n: int, steps: int=100) -> np.ndarray:
#         samples = np.zeros((n, self.rbms[0].a.shape[0]))
#         last_rbm_samples = self.rbms[-1].generate(n, steps)

#         for i in range(n):
#             sample = last_rbm_samples[i]
#             for rbm in self.rbms[-2::-1]:
#                 sample = rbm.sortie_entree(sample)
#                 sample = np.random.binomial(1, sample)
#             samples[i] = sample
        
#         return samples

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
    

# if __name__ == "__main__":
#     from load_data import lire_alpha_digit
#     characters = ["m"]
#     images = lire_alpha_digit(characters)
#     dbn = DBN([320, 128, 128, images.shape[-1]])
#     dbn.train(images, epochs=10000, batch_size=16, eps=0.001)
#     generated_images = dbn.generate(20)
#     generated_images = generated_images.reshape(20, 20, 16)
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 5))
#     for i in range(20):
#         plt.subplot(4, 5, i+1)
#         plt.imshow(generated_images[i], cmap="gray")
#         plt.axis("off")
#     plt.show()   



# if __name__ == "__main__":
#     # Train the DBN on the binary alpha digit dataset
#     from load_data import lire_alpha_digit

#     characters = ["m"]
#     images = lire_alpha_digit(characters)

#     dbn = DBN([320, 128, 128, images.shape[-1]])
#     dbn.train(images, epochs=200, batch_size=16, eps=0.001)

#     # Generate new images
#     generated_images = dbn.generate(20)
#     generated_images = generated_images.reshape(20, 20, 16)

#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(10, 5))
#     for i in range(20):
#         plt.subplot(4, 5, i+1)
#         plt.imshow(generated_images[i], cmap="gray")
#         plt.axis("off")
#     plt.show()
