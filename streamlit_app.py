
import streamlit as st
import importlib
from typing import TYPE_CHECKING
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
st.set_page_config(page_title='CIFAR-10 Classifier', layout='centered')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- MODEL DEFINITION (must match training Net) ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(path='models/cifar_net.pth'):
    model = Net()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Try to load model if it exists; otherwise keep None and show instructions in the UI
def try_load_model(path='models/cifar_net.pth'):
    if os.path.exists(path):
        return load_model(path)
    return None

model = try_load_model()

st.title('CIFAR-10 Image Classifier')
uploaded = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

if uploaded is not None:
    if model is None:
        st.error('Model file `models/cifar_net.pth` not found. Run the training notebook to create the model, or place a trained `cifar_net.pth` in the `models/` folder.')
    else:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption='Uploaded image', use_column_width=True)
        input_tensor = transform(image).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).numpy().squeeze()
            top_idx = np.argmax(probs)
            st.write(f'Predicted: **{classes[top_idx]}**')
            st.write('Class probabilities:')
            for i, c in enumerate(classes):
                st.write(f'{c}: {probs[i]:.4f}')
else:
    st.write('Upload an image to classify.')
