import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from model_experimentation.models import *

writer = SummaryWriter(log_dir="runs/test_embeddings")


def log_embeddings(model, data_loader, device, num_samples=500):
    """
    Logs embeddings for visualization in TensorBoard.

    Works with:
    - CNNRULClassifier (1D CNN)
    - CNNRUL2DClassifier (2D CNN)
    - HybridCNNClassifier (1D + 2D Hybrid CNN)
    - ComplexHybridCNNClassifier (Advanced 1D + 2D Hybrid CNN)

    Parameters:
    - model: PyTorch model.
    - data_loader: DataLoader providing test/validation data.
    - device: 'cuda' or 'cpu'.
    - num_samples: Max number of samples to log.
    """
    model.eval()
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if isinstance(model, CNNRULClassifier):
                print("got here")
                inputs = inputs.permute(0, 2, 1)

            # Use feature_extractor()
            features = model.feature_extractor(inputs)
            embeddings.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            # Stop once we've collected enough samples
            if len(embeddings) * inputs.shape[0] >= num_samples:
                break

    embeddings = np.concatenate(embeddings, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # Log embeddings to TensorBoard
    writer.add_embedding(mat=embeddings, metadata=labels_list, tag="Model_Embeddings")
    writer.close()
