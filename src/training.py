import tensorflow as tf
import keras as K
import numpy as np
from sklearn.model_selection import train_test_split
from ae_motor_module import MotorModuleNNAE
from argparse import ArgumentParser
from typing import Optional
import os

def _train_motor_module_ae(
    data_dir: str,
    columns: Optional[list[int]],
    max_modules: Optional[int],
    output_folder: str,
    output_format: str
):
    pass

if __name__ == "__main__":
    parser = ArgumentParser(description="Train Motor Module Autoencoder on EMG data.")
    parser.add_argument('data_dir', type=str, help="Path to the EMG data file (CSV format).")
    parser.add_argument('--columns', type=int, nargs='+', required=False, help="List of column indices to use as input features. If not specified, all columns will be used.")
    parser.add_argument('--max_modules', type=int, required=False, help="Maximum number of motor modules (latent dimensions) to extract. If not specified, defaults to the number of input muscles.")
    parser.add_argument('--out', type=str, required=False, help="Folder to save the trained models. Defaults to 'results' in the current directory.", default='./results')
    parser.add_argument('--outfmt', type=str, choices=['folder', 'tar'], default='folder', help="Format to save the models: 'folder' to save in a directory for each input file. 'tar' to save as a single .tar file.")
    
    args = parser.parse_args()
    print(args)
    _train_motor_module_ae(
        data_dir=args.data_dir,
        columns=args.columns,
        max_modules=args.max_modules,
        output_folder=args.output_folder,
        output_format=args.output_format
    )