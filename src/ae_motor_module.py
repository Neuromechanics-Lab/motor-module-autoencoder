import keras as K
import tensorflow as tf
import numpy as np
from typing import Union, Literal

class MotorModuleNNAE(K.Model):
    def __init__(self, n_muscles: int, latent_dim: int):
        """Autoencoder model for extracting motor modules (synergies) from EMG data.

        Args:
            n_muscles (int): Number of input muscles (features).
            latent_dim (int): Dimensionality of the latent synergy space.
        """
        super(MotorModuleNNAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_muscles = n_muscles
        self.encoder = K.Sequential([
            K.layers.Dense(latent_dim, activation='relu', name='latent', kernel_constraint=K.constraints.NonNeg())
        ], name='encoder')
        self.decoder = K.Sequential([
            K.layers.Dense(n_muscles, activation='relu', name='output', kernel_constraint=K.constraints.NonNeg(), kernel_regularizer=K.regularizers.L2())
        ], name='decoder')

    def call(self, x):
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def modules(self):
        """Get the learned motor modules (synergies) from the decoder weights."""
        output_weights = self.decoder.get_layer('output').get_weights()[0]
        return output_weights
    
    def synergies(self):
        """Get the learned motor modules (synergies) from the decoder weights. Identical to calling `modules()`."""
        return self.modules()
    
    def activations(self, x):
        """Get the activations for each motor module (synergy) given input EMG data `x`."""
        return self.encoder(x).numpy()
    
    def no_bias(self)-> K.Model:
        model_copy = self.dupe()
        model_copy.build(input_shape=(None, self.n_muscles))
        model_copy._strip_bias()
        return model_copy
    
    def bias_only(self) -> K.Model:
        model_copy = self.dupe()
        model_copy.build(input_shape=(None, self.n_muscles))
        model_copy._strip_weights()
        return model_copy

    def _strip_bias(self):
        """Set all biases in encoder and decoder layers to zero. Can be used after training to evaluate without bias terms."""
        for layer in self.encoder.layers:
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.assign(tf.zeros_like(layer.bias))
        for layer in self.decoder.layers:
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.assign(tf.zeros_like(layer.bias))

    def _strip_weights(self):
        """Set all weights in encoder and decoder layers to zero. Can be used after training to evaluate without weight terms."""
        for layer in self.encoder.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                layer.kernel.assign(tf.zeros_like(layer.kernel))
        for layer in self.decoder.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                layer.kernel.assign(tf.zeros_like(layer.kernel))

    def seed_from(self, seeding: Union[K.Model, tuple, dict]):
        """Seed the autoencoder weights from a pre-trained model or given weights."""
        if isinstance(seeding, K.Model):
            enc_w, enc_b = seeding.encoder.get_layer('encoder').get_weights()
            dec_w, dec_b = seeding.decoder.get_layer('decoder').get_weights()
        elif isinstance(seeding, tuple):
            (enc_w, enc_b), (dec_w, dec_b) = seeding
        elif isinstance(seeding, dict):
            enc_w = seeding['encoder_weights']
            enc_b = seeding['encoder_bias']
            dec_w = seeding['decoder_weights']
            dec_b = seeding['decoder_bias']
        assert (enc_w.shape <= self.encoder.get_layer('encoder').get_weights()[0].shape and
                dec_w.shape <= self.decoder.get_layer('decoder').get_weights()[0].shape), \
                "Seed model must have latent and output layer dimensions less than or equal to the new model."
        # Add a random vector if the dimensions don't match
        enc_seed_weights = np.random.rand(*self.encoder.get_layer('encoder').get_weights()[0].shape)
        enc_seed_weights[:enc_w.shape[0], :enc_w.shape[1]] = enc_w
        enc_seed_biases = np.zeros_like(self.encoder.get_layer('encoder').get_weights()[1])
        enc_seed_biases[:enc_b.shape[0]] = enc_b
        self.encoder.get_layer('encoder').set_weights([enc_seed_weights, enc_seed_biases])
        dec_seed_biases = np.zeros_like(self.decoder.get_layer('decoder').get_weights()[1])
        dec_seed_biases[:dec_b.shape[0]] = dec_b
        dec_seed_weights = np.random.rand(*self.decoder.get_layer('decoder').get_weights()[0].shape)
        dec_seed_weights[:dec_w.shape[0], :dec_w.shape[1]] = dec_w
        self.decoder.get_layer('decoder').set_weights([dec_seed_weights, dec_seed_biases])

    def self_fit(self, X, val_data=None, verbose=0):
        """Fit the autoencoder to the input data `X`. Essentially an alias for `model.fit(X, X)`, but will also build and compile the model if not already done.

        Args:
            X (np.array): Input EMG data of shape (n_samples, n_muscles).
            val_data (np.array, optional): Optional validation data of shape (n_val_samples, n_muscles). If provided, early stopping will be enabled. Defaults to None.
            verbose (int, optional): Verbosity mode. Defaults to 0.
        """
        self.compile(optimizer='adam', loss='mse')
        self.build(input_shape=(None, self.n_muscles))
        if val_data is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            self.fit(X, X, epochs=100, shuffle=False, batch_size=32, verbose=verbose, validation_data=(val_data, val_data), callbacks=[early_stopping])
        else:
            self.fit(X, X, epochs=100, shuffle=False, batch_size=32, verbose=verbose)

    def save_model(self, filepath):
        """Save the entire model weights. The architecture should be reloaded by creating a new object.

        Args:
            filepath (str): Path to save the model.
        """
        self.save_weights(filepath)

    def load_model(self, filepath):
        """Load the entire model weights.

        Args:
            filepath (str): Path to load the model from.
        """
        self.build(input_shape=(None, self.n_muscles))
        self.load_weights(filepath)


    def dupe(self) -> K.Model:
        """Create a copy of the model"""
        model_copy = MotorModuleNNAE(self.n_muscles, self.latent_dim)
        model_copy.build(input_shape=(None, self.n_muscles))
        model_copy.encoder.set_weights(self.encoder.get_weights())
        model_copy.decoder.set_weights(self.decoder.get_weights())
        return model_copy

    def reconstruct(self, X:Union[np.ndarray, tf.Tensor], using_modules:Union[list[int], Literal["all"]]="all", using_bias:bool=True):
        """Reconstruct the input data using the specified modules.

        Args:
            X (Union[np.ndarray, tf.Tensor]): Input EMG data of shape (n_samples, n_muscles).
            using_modules (Union[list[int], Literal["all"]]): List of modules indicies (0 indexed) to use for reconstruction or "all" to use all modules.
            using_bias (bool): Whether to use the bias terms in the reconstruction. Defaults to True.
        """

        model_to_use = self.dupe()
        if using_modules != "all":
            model_to_use = self.dupe()
            # Zero out weights for unused modules -> i.e. the weights from the latent layer to the output layer, corresponding to unused latent space dimensions
            output_layer = model_to_use.decoder.get_layer('output')
            weights, biases = output_layer.get_weights()
            for i in range(weights.shape[0]):
                if i not in using_modules:
                    weights[i, :] = 0
            output_layer.set_weights((weights, biases))
        if not using_bias:
            model_to_use._strip_bias()
        return model_to_use(X).numpy()

if __name__ == "__main__":
    SELECTED_MUSCLES = [0,4,7]
    MAX_SYNS = 6
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
    SUBJECT = "AB10"
    train_data = np.loadtxt(f"D:\\Dropbox-GT\\GaTech Dropbox\\ME-DboxMgmt-Young-Admins\\Siddharth Nathella\\Projects\\MuscleSynergyAutoencoder\\Code\\results\\camargo3\\ae\\{SUBJECT}\\train_data.csv", delimiter=",")
    for n_syn in range(1, MAX_SYNS+1):
        ae_model = MotorModuleNNAE(n_muscles=11, latent_dim=n_syn)
        ae_model.load_model(f"D:\\Dropbox-GT\\GaTech Dropbox\\ME-DboxMgmt-Young-Admins\\Siddharth Nathella\\Projects\\MuscleSynergyAutoencoder\\Code\\results\\camargo3\\ae\\{SUBJECT}\\{n_syn}\\AE.h5")
        recon = ae_model.reconstruct(train_data)
        for i, m in enumerate(SELECTED_MUSCLES):
            # First plot the complete reconstruction
            plt.subplot(MAX_SYNS, len(SELECTED_MUSCLES), plotnum)
            plt.plot(train_data[:1500, m], label="Original", color="k", linestyle='-.', linewidth=2)
            plt.plot(recon[:1500, m], label="Reconstructed", color="grey", alpha=0.7, linestyle='-.', linewidth=2)
            plt.xticks([])
            plt.yticks([])
            plt.ylim((0, 1))
            # Remove spines
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            for j in range(n_syn): # Plot the reconstruction from each synergy
                recon_j = ae_model.reconstruct(train_data, using_modules=[j], using_bias=False)
                plt.plot(recon_j[:1500, m], label=f"Syn {j+1}", alpha=0.7, linewidth=2, color=color_list[j])
            # Show bias only reconstruction
            recon_bias = ae_model.reconstruct(train_data, using_modules=[], using_bias=True)
            plt.plot(recon_bias[:1500, m], label="Bias Only", color="r", alpha=0.7, linestyle=':', linewidth=2)
            plotnum += 1
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig("ae_reconstructions.pdf", bbox_inches='tight')