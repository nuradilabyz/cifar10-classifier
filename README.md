# CIFAR-10 Classifier

This repository contains an end-to-end example project that trains a simple CNN on CIFAR-10 using PyTorch and serves a Streamlit app for image classification.

Files:
- `cifar10_training.ipynb` - Jupyter notebook for data loading, training, evaluation, and saving the model to `models/cifar_net.pth`.
- `streamlit_app.py` - Streamlit application that loads the saved model and classifies uploaded images.
- `requirements.txt` - required Python packages.

Quick start:
1. Create a virtual environment and activate it.
2. Install dependencies:

   pip install -r requirements.txt

3. Train (optional):
   - Open `cifar10_training.ipynb` and run cells to train and save the model. The training example uses 2 epochs for speed; increase for better accuracy.

4. Run Streamlit app:

   streamlit run streamlit_app.py

Model and data will be stored under `models/` and `data/` respectively.
