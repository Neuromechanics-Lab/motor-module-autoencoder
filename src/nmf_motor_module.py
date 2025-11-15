import numpy as np
from typing import List, Union
class NMFMotorModule:
    def __init__(self, n_muscles, n_modules):
        self.n_muscles = n_muscles
        self.n_modules = n_modules
        self.weights = None
        self.activations = None
        self.std_dev = None
        self.mean = None

    def fit(self, muscle_activations):
        raise NotImplementedError("This method will be implemented soon.")

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
            reconstructions = self.weights @ activations
        else:
            reconstructions = self.weights[:, using_modules] @ activations[using_modules, :]
        return reconstructions
    

if __name__ == "__main__": 
    SELECTED_MUSCLES = [0,4,7]
    MAX_SYNS = 6
    SUBJECT = "AB10"
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plotnum = 1
    color_list = [
        'xkcd:sea blue',
        'xkcd:tangerine',
        'xkcd:brick red',
        'xkcd:emerald',
        'xkcd:deep sky blue',
        'xkcd:purple',
    ]
    train_data = np.loadtxt(f"D:\\Dropbox-GT\\GaTech Dropbox\\ME-DboxMgmt-Young-Admins\\Siddharth Nathella\\Projects\\MuscleSynergyAutoencoder\\Code\\results\\camargo3\\nmf\\{SUBJECT}\\train_data.csv", delimiter=",")

    for n_syn in range(1, MAX_SYNS+1):
        W = np.loadtxt(f"D:\\Dropbox-GT\\GaTech Dropbox\\ME-DboxMgmt-Young-Admins\\Siddharth Nathella\\Projects\\MuscleSynergyAutoencoder\\Code\\results\\camargo3\\nmf\\{SUBJECT}\\{n_syn}\\W.csv", delimiter=",")
        C = np.loadtxt(f"D:\\Dropbox-GT\\GaTech Dropbox\\ME-DboxMgmt-Young-Admins\\Siddharth Nathella\\Projects\\MuscleSynergyAutoencoder\\Code\\results\\camargo3\\nmf\\{SUBJECT}\\{n_syn}\\H.csv", delimiter=",")
        if n_syn == 1:
            W = W.reshape(1, -1)
            C = C.reshape(-1, 1)
        print(W.shape, C.shape)
        nmf_model = NMFMotorModule(n_muscles=11, n_modules=n_syn)
        nmf_model.load_weights(W.T, C.T)
        recon = nmf_model.reconstruct()
        for i, m in enumerate(SELECTED_MUSCLES):
            plt.subplot(MAX_SYNS, len(SELECTED_MUSCLES), plotnum)
            # First plot the complete reconstruction
            plt.plot(train_data[:1500, m], label="Original", color="k", linestyle='-.', linewidth=2)
            plt.plot(recon[m, :1500], label="Reconstructed", color="grey", alpha=0.7, linewidth=2)
            plt.xticks([])
            plt.yticks([])
            plt.ylim((0, 1))
            # Remove spines
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            for j in range(n_syn): # Plot the reconstruction from each synergy
                recon_j = nmf_model.reconstruct(using_modules=[j])
                plt.plot(recon_j[m, :1500], label=f"Syn {j+1}", alpha=0.7, linewidth=2, color=color_list[j])
            plotnum += 1
    plt.tight_layout()
    # plt.show()
    fig = plt.gcf()
    fig.savefig("nmf_reconstructions.pdf", bbox_inches='tight')
            