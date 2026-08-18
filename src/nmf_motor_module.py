from copy import deepcopy
import numpy as np
from typing import List, Union
from sklearn.decomposition import NMF
class NMFMotorModule:
    def __init__(self, n_muscles, n_modules):
        self.n_muscles = n_muscles
        self.n_modules = n_modules
        self.weights = None
        self.activations = None
        self.std_dev = None
        self.mean = None
        self.W_init = None
        self.H_init = None
        self.recons = None

    def seed_from(self, other):
        pass

    def fit(self, muscle_activations):
        emg_data = deepcopy(muscle_activations)
        # Clip values below zero
        emg_data[emg_data < 0] = 0

        self.mean = np.mean(emg_data, axis=0)
        self.std_dev = np.std(emg_data, axis=0)

        unit_norm_emg = emg_data / self.std_dev

        if self.W_init is None and self.H_init is None:
            nmf_model = NMF(
                n_components=self.n_modules,
                init='random',
                solver='mu',
                max_iter=100000
            )
            W = nmf_model.fit_transform(unit_norm_emg)
        else:
            nmf_model = NMF(
                n_components=self.n_modules,
                init='custom',
                solver='mu',
                max_iter=100000
            )
            W = nmf_model.fit_transform(unit_norm_emg, W=self.W_init, H=self.H_init)

        H = nmf_model.components_
        # Re-scale
        H = np.multiply(H, self.std_dev)
        m = np.max(H, axis=1)
        H = np.divide(H.T, m).T
        W = np.multiply(W, m)
        recon = np.matmul(W, H)
        self.weights = H
        self.activations = W
        self.recons = recon

    # def modules(self):
    #     return self.weights

    # def synergies(self):
    #     return self.modules()
    
    # def activations(self):
    #     return self.activations

    # def reconstruct(self):
        # return self.recons

    def self_fit(self, X):
        return self.fit(X)

    def load_weights(self, weights, activations):
        if weights.shape != (self.n_muscles, self.n_modules):
            raise ValueError(f"weights must be of shape ({self.n_muscles}, {self.n_modules})")
        if activations.shape[0] != self.n_modules:
            raise ValueError(f"activations must have {self.n_modules} rows")
        self.weights = weights
        self.activations = activations

    def reconstruct(self, activations=None, using_modules:Union[str, List[int]]="all"):
        if activations is None:
            activations = self.activations
        if using_modules == "all":
            reconstructions = activations @ self.weights
        else:
            reconstructions = self.weights[:, using_modules] @ activations[using_modules, :]
        return reconstructions
            