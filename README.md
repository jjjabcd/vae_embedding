# Generative Model Final Project

## Codebase
[vae-embeddings](https://github.com/znavoyan/vae-embeddings)

## Improving VAE based molecular representations for compound property prediction
This repository contains training and inference codes for CVAE architechture designed for downstream task, that are introduced in [Improving VAE based molecular representations for compound property prediction](https://arxiv.org/abs/2201.04929)

In the paper we propose a simple method to improve chemical property prediction performance of machine learning models by incorporating additional information on correlated molecular descriptors in the representations learned by variational autoencoders.

## Installation
For running the codes, you need to install a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment with all the required packages by running the following commands.

### Environment Setup for CVAE (TensorFlow 2.14 + GPU Support)

The original code was designed for TensorFlow 1.x, but we have updated it to work with TensorFlow 2.14 and modern GPU setups. Here's the complete setup process:

#### 1. Create Conda Environment
```bash
# Create environment based on DECIMER setup for better compatibility
conda create -n chemvae python=3.9 -y
conda activate chemvae

# Install core packages
conda install -c conda-forge cudatoolkit=11.8 tensorflow-gpu=2.14 keras-preprocessing scikit-learn=1.6 matplotlib seaborn pillow deepsmiles=1.0.1 jupyter rdkit=2024.09.2 tqdm pyyaml=5.4.1 -y

# Install CUDA compiler (required for some operations)
conda install -c nvidia cuda-nvcc=12.4.131 -y
```

#### Alternative Installation (Using environment.yml)
```bash
# Create environment from the updated environment file
conda env create -f environment_cvae.yml
conda activate chemvae

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

**Note:** The `environment_cvae.yml` file has been updated to match the successful installation process described above. This provides an alternative way to set up the environment in a single command.

#### 2. Verify GPU Setup
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

#### 3. Code Compatibility Updates
The code has been updated for TensorFlow 2.x compatibility:
- Updated import statements (`from keras` → `from tensorflow.keras`)
- Fixed optimizer parameters (`lr` → `learning_rate`)
- Resolved XLA/JIT compilation issues for modern GPUs
- Updated RNN layer implementations
- Fixed TensorFlow 2.x API changes

## Training
For each specific downstream task, e.g. solubility prediction (LogS), the training process consists of three steps:
1. Train variational autoencoder (CVAE) 
2. Extract the molecular embeddings from the trained VAE
3. Train another neural network for the downstream task

### Step 1: Training of VAE
For the training of VAE we are using 250k excerpt of ZINC dataset placed in `data/zinc` folder.

**CVAE (Updated for TensorFlow 2.14)**
```bash
cd chemical_vae
conda activate chemvae
python -m chemvae.train_vae_new -d models/zinc_logp_196/
```

**Important Notes for CVAE Training:**
- Ensure you're in the correct directory (`chemical_vae`) before running
- The conda environment (`chemvae`) must be activated
- `-d` specifies the model's directory, which should include `exp.json` file with all the parameters for training the model
- GPU training is now supported with RTX 4090 and other modern GPUs
- Training will automatically use GPU if available, with CPU fallback
- XLA/JIT compilation is automatically disabled to prevent compatibility issues

### Step 2: Extracting molecular embeddings
In this step, by already having the pre-trained VAE model, we can encode the molecules from downstream task's dataset into high dimensional embeddings. 

**Important:** Make sure you're in the root directory (`vae_embedding/`) before running the command.

The code below shows an example of getting embeddings for Solubility prediction dataset using CVAE trained with MolLogP property predictor:
  ```bash
  # Set environment variables for GPU optimization
  # IMPORTANT: Replace {username} with your actual username and adjust the path to your conda environment
  export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/{username}/miniconda3/envs/chemvae
  
  # To find your conda environment path, you can use:
  # conda info --envs
  # or
  # echo $CONDA_PREFIX (when chemvae environment is activated)
  
  # Extract molecular embeddings using the trained VAE model
  python src/fingerprints/vae.py --input data/logS/processed/final_logS_6789.csv --model_dir chemical_vae/models/zinc_logp_196/ --output data/logS/processed_with_cvae/final_logS_cvae_logp_196.csv
  ```

**Parameters:**
- `--input`: Path to downstream task's dataset
- `--model_dir`: Path to variational autoencoder model trained during Step 1
- `--output`: Path where the dataset enriched with embeddings will be saved

  **Important Notes:**
  - The script automatically handles molecules with unsupported elements and will skip them with appropriate error messages.
  - **XLA Environment Variable**: You must replace `{username}` in the export command with your actual username. Common conda installation paths:
    - Linux/Mac: `/home/{username}/miniconda3/envs/chemvae` or `/home/{username}/anaconda3/envs/chemvae`
    - To find your exact path: run `conda info --envs` or `echo $CONDA_PREFIX` (when chemvae is activated)
  - If you encounter XLA/JIT compilation errors, this environment variable resolves CUDA libdevice path issues.

### Step 3: Training of model for downstream task
After extracting molecular embeddings, we can now train the model for downstream task. We support three different model types: ResNet (1D CNN), MLP (Multi-Layer Perceptron), and LR (Linear Regression).

**Important:** Make sure you're in the root directory (`vae_embedding/`) before running the command.

The following code shows examples of training different models for Solubility (LogS) prediction task:

**MLP Model (Recommended for quick testing):**
```bash
# Train MLP model using VAE embeddings
python src/train.py --property logS --data data/logS/processed_with_cvae/final_logS_cvae_logp_196_6679.csv --save_dir models/cv10_logS_6679_cvae_emb_logp_196 --feature vae_emb --fold_indices_dir data/logS/fold_indices_cvae/ --model MLP
```

**Linear Regression Model (Fastest baseline):**
```bash
python src/train.py --property logS --data data/logS/processed_with_cvae/final_logS_cvae_logp_196_6679.csv --save_dir models/cv10_logS_6679_cvae_emb_logp_196_lr --feature vae_emb --fold_indices_dir data/logS/fold_indices_cvae/ --model LR
```

**ResNet Model (Best performance, requires TensorFlow 2.x updates):**
```bash
# Note: ResNet model requires additional TensorFlow 2.x compatibility fixes
python src/train.py --property logS --data data/logS/processed_with_cvae/final_logS_cvae_logp_196_6679.csv --save_dir models/cv10_logS_6679_cvae_emb_logp_196_resnet --feature vae_emb --fold_indices_dir data/logS/fold_indices_cvae/ --model ResNet
```
**Parameters:**
- `--property`: Name of the downstream task ('logS', 'logBB', or 'logD')
- `--data`: Path to the downstream task's dataset (with VAE embeddings)
- `--save_dir`: Directory where training results and models will be saved
- `--feature`: Input representation type ('vae_emb' for VAE embeddings)
- `--fold_indices_dir`: Directory for cross-validation fold indices
- `--model`: Model type ('ResNet', 'MLP', or 'LR')

```
--fold_num: number of folds for cross validation, default = 10
--repeat_folds: number of times cross validation is repeated, default = 1
--start_fold: specifies from which fold the training should start/continue, in case the training is interrupted, default = 1
--epochs: number of epochs for ResNet, if not specified, the default values for LogS, LogD and LogBB are 2000, 1500 and 85 respectively
--learning_rate: learning rate for ResNet 
--batch_size: batch size for ResNet 
--l2_wd: L2 weight decay regularization for ResNet 
--mlp_max_iter: maximum number of iterations for MLP
```

## Evaluation
After training the models, you can evaluate their performance and get predictions for each fold by running the `test.py` script.

**Important:** Make sure you're in the root directory (`vae_embedding/`) before running the command.

### Evaluate MLP Model
```bash
# Evaluate the trained MLP model
python src/test.py --experiment models/cv10_logS_6679_cvae_emb_logp_196 --model MLP
```

### Evaluate Linear Regression Model
```bash
python src/test.py --experiment models/cv10_logS_6679_cvae_emb_logp_196_lr --model LR
```

### Evaluate ResNet Model (when available)
```bash
# Note: ResNet evaluation requires TensorFlow 2.x compatibility updates
python src/test.py --experiment models/cv10_logS_6679_cvae_emb_logp_196_resnet --model ResNet
```

**Parameters:**
- `--experiment`: Directory path containing the trained model(s) and configuration
- `--model`: Model type to evaluate ('ResNet', 'MLP', or 'LR')


## Troubleshooting

### Common Issues and Solutions

#### 1. TensorFlow 1.x Compatibility Issues
**Problem:** Original code designed for TensorFlow 1.x may not work with modern setups.
**Solution:** Use the updated environment setup with TensorFlow 2.14 as described above.

#### 2. GPU Not Recognized
**Problem:** GPU not being utilized during training.
**Solution:** 
- Verify CUDA installation: `nvidia-smi`
- Check TensorFlow GPU detection: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Ensure proper CUDA toolkit version (11.8) is installed

#### 3. XLA/JIT Compilation Errors
**Problem:** Errors related to `libdevice.10.bc` or XLA compilation.
**Solution:** 
- Set the XLA environment variable with your correct conda path:
  ```bash
  export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/your/conda/envs/chemvae
  ```
- To find your conda environment path:
  ```bash
  conda info --envs
  # or when chemvae is activated:
  echo $CONDA_PREFIX
  ```
- The updated code automatically disables XLA/JIT compilation for most operations to prevent these issues.

#### 4. Module Import Errors
**Problem:** `ImportError` or module not found errors.
**Solution:** 
- Ensure you're in the correct directory (`chemical_vae` for CVAE)
- Activate the conda environment: `conda activate chemvae`
- Use the module import format: `python -m chemvae.train_vae_new`

#### 5. Memory Issues
**Problem:** Out of memory errors during training.
**Solution:** 
- GPU memory growth is automatically enabled
- Reduce batch size in `exp.json` if needed
- Monitor GPU memory usage with `nvidia-smi`

### Performance Notes
- Training on RTX 4090: ~39 seconds per epoch for ZINC dataset
- CPU training is significantly slower but still functional
- Modern GPUs (RTX 30xx/40xx series) are fully supported

## License
Apache License Version 2.0
