# motor-module-autoencoder
Code for computing motor modules from electromyography (EMG) measured muscle activity, using an autoencoder

## Quickstart
In order to use this package, please make sure that you have access to a GPU enabled runtime. This can be done easily with a conda environment, with the following command to install the cuda tools.
```console
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Alternatively, a google colab notebook will likely automatically have this available.

To install this to your python environment, you can directly install with 
```console
pip install git+https://github.com/siddn/motor-module-autoencoder
```

or include this in the first cell of your google colab notebook.
```python
!pip install git+https://github.com/siddn/motor-module-autoencoder
```


### How to use
Operation can be done in one of two ways.
1) You can directly interface with the neural network object designed for the motor module extraction. This is especially useful if you want to do some more in depth analysis of all the parts that come out of the autoencoder, or if you would like to integrate it into your pre-existing pipelines in a way that a single function call is inconvenient for. There are some convenience/utility functions that are provided for pre-processing and analysis steps that you may find helpful.
```python
from motor_module_autoencoder import MotorModuleNNAE
mm_model = MotorModuleNNAE(n_muscles=10, n_modules=3)

# OPTIONAL seeding of the model with either a) another MotorModuleNNAE object, OR a tuple of tuples, where the first entry is the weights and biases of the encoder and the second is the weights and biases of the decoder, OR a dict with the keys 'encoder_weight', 'encoder_bias', 'decoder_weight', 'decoder_bias'.
mm_model.seed_from(old_model)
# or nn.model.seed_from(((np.zeros(10, 3), np.zeros(10,)), (np.zeros(10, 3), np.zeros(10,))))
# or nn

mm_model.self_fit(your_data) # You can optionally provide validation data to enable early stopping.
# Analysis
motor_modules = mm_model.modules()
activations = mm_model.activation(your_data)
new_activations = mm_model.activations(some_new_data) # Motor modules from some new unseen data
reconstruction = mm_model(you_data)
# Removing the weights or bias for fine grained analysis. These should only be called AFTER `.self_fit`.
no_bias_mm_model = mm_model.no_bias()
bias_only_mm_model = mm_model.bias_only()
# Saving and reloading the pure model can be done like any typical keras model. This can be helpful as saving each individual part can be inconvenient to recreate the model later. The modules and activations can be saved seperately in CSV or text files for seperate analysis if desired
mm_model.save_model("path/to/save/model.h5")
np.savetxt("path/to/modules.csv", motor_modules, delimiter=',')
np.savetxt("path/to/activations.csv", activations, delimiter=',')
# Reload
reloaded_model = MotorModuleNNAE
reloaded_model.load_model("path/to/save/model.h5")
```

2) There is a command line interface available. If you have no pre-existing piplines and wish to operate on a directory of pre-existing files, this may be a valuable option. Your directory must be structured in a subject/task/repetition format. The output directory will mimic the input directory, with a folder (or tarball) for each subject containing the model weights and the Module weights, biases and activations based on the input data. This option has some fairly strict pre-configured settings, and will not output the bias-only/no-bias cases. You can post-hoc get these by reloading the models, but that has to be written using the above option. The command will be available after installing the package. If you are using the pure files, you need to target the function to run with `python -m src/training.py`. The columns are 0 indexed, so if a time column exists in your data make sure to start your columns count at 1.
```console
motor-module-ae path/to/data --columns 1 2 3 4 5 6 7 8 9 10 --max-modules 4 --out path/to/results --ofmt folder
```

## MATLAB integration - Work in progress
The matlab integration uses the built-in python caller in matlab. This allows us to use all the ML dependencies like tensorflow and scikit-learn without having to start from the ground up. In order to use this in matlab, you will need to direct matlab to where your python runtime is. This will not work if you use Colab and do not have a local python environment.
1) Follow the aboove quickstart instructions up until you have a conda environment that can effectively import the required libraries. Run the following command in your terminal to check that everything is good to go
```console
python -c "import tensorflow as tf; import sklearn; print(f'GPU devices found: {tf.config.list_physical_devices(\"GPU\")}');"
```
If this does not succesfully print with a GPU, the code may still work on the CPU but it will train slower.
