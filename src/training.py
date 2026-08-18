import tensorflow as tf
import keras as K
import numpy as np
from sklearn.model_selection import train_test_split
from ae_motor_module import MotorModuleNNAE
from nmf_motor_module import NMFMotorModule
from argparse import ArgumentParser
from typing import Optional
import os

def _train_motor_module_ae(
    data_dir: str,
    max_modules: Optional[int],
    output_folder: str,
    random_state: Optional[int] = 42
):
    files_ = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for file in files_:
        file_path = os.path.join(data_dir, file)
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        n_muscles = data.shape[1]
        X_train, X_val = train_test_split(data, test_size=0.2, random_state=random_state)
        for n_modules in range(1, max_modules + 1):
            # Create the autoencoder model
            model = MotorModuleNNAE(n_muscles=n_muscles, latent_dim=n_modules)
            model.compile(optimizer='adam', loss='mse')
            model.build(input_shape=(None, n_muscles))
            # Train the model
            model.self_fit(X_train, val_data=X_val, verbose=1)
            # Save the model weights
            results_dir = os.path.join(output_folder, f"{file.split('.')[0]}", str(n_modules))
            os.makedirs(results_dir, exist_ok=True)
            reconstructions = model.reconstruct(X_train)
            modules = model.modules()
            bias = model.bias()
            activations = model.activations(X_train)
            # Save the reconstructions, modules, and activations
            model.save_model(os.path.join(results_dir, "AE.h5"))
            np.savetxt(os.path.join(results_dir, "AE_recon.csv"), reconstructions, delimiter=',')
            np.savetxt(os.path.join(results_dir, "AE_modules.csv"), modules, delimiter=',')
            np.savetxt(os.path.join(results_dir, "AE_activations.csv"), activations, delimiter=',')
            np.savetxt(os.path.join(results_dir, "AE_bias.csv"), bias, delimiter=',')
            np.savetxt(os.path.join(results_dir, "AE_train_data.csv"), X_train, delimiter=',')
            np.savetxt(os.path.join(results_dir, "AE_val_data.csv"), X_val, delimiter=',')

def _train_motor_module_nmf(
    data_dir: str,
    max_modules: Optional[int],
    output_folder: str,
    random_state: Optional[int] = 42
):
    files_ = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for file in files_:
        data = np.loadtxt(os.path.join(data_dir, file), delimiter=',', skiprows=1)
        n_muscles = data.shape[1]
        for n_modules in range(1, max_modules + 1):
            # Create the NMF model
            nmf_model = NMFMotorModule(n_muscles=n_muscles, n_modules=n_modules)
            # Fit the model
            nmf_model.fit(data)
            # Save the model weights
            results_dir = os.path.join(output_folder, f"{file.split('.')[0]}", str(n_modules))
            os.makedirs(results_dir, exist_ok=True)
            reconstructions = nmf_model.recons
            modules = nmf_model.weights
            activations = nmf_model.activations
            # Save the reconstructions, modules, and activations
            np.savetxt(os.path.join(results_dir, "NMF_recon.csv"), reconstructions, delimiter=',')
            np.savetxt(os.path.join(results_dir, "NMF_modules.csv"), modules, delimiter=',')
            np.savetxt(os.path.join(results_dir, "NMF_activations.csv"), activations, delimiter=',')
            np.savetxt(os.path.join(results_dir, "NMF_train_data.csv"), data, delimiter=',')

if __name__ == "__main__":
    parser = ArgumentParser(description="Train Motor Module Autoencoder on EMG data.")
    parser.add_argument('data_dir', type=str, help="Path to the EMG data file (CSV format).")
    parser.add_argument('--mode', type=str, choices=['ae', 'nmf'], required=True, help="Choose the training mode: 'ae' for Autoencoder or 'nmf' for Non-negative Matrix Factorization.")
    parser.add_argument('--max_modules', type=int, required=False, help="Maximum number of motor modules (latent dimensions) to extract. If not specified, defaults to the number of input muscles.")
    parser.add_argument('--out', type=str, required=False, help="Folder to save the trained models. Defaults to 'results' in the current directory.", default='./results')
    parser.add_argument('--random_state', type=int, required=False, help="Random state for reproducibility. Defaults to 42.", default=42)

    args = parser.parse_args()
    if args.mode == 'ae':
        _train_motor_module_ae(
            data_dir=args.data_dir,
            max_modules=args.max_modules,
        output_folder=args.out,
        random_state=args.random_state
    )
    elif args.mode == 'nmf':
        _train_motor_module_nmf(
            data_dir=args.data_dir,
            max_modules=args.max_modules,
            output_folder=args.out,
            random_state=args.random_state
        )