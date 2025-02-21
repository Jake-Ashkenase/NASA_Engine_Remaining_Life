import torch
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
data_structures_path = os.path.join(parent_dir, "data_structures")
sys.path.append(data_structures_path)
from data_structures import EarlyStopper
from .regression import CNNRULRegression
from .classification import CNNRULClassifier

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="runs/test_embeddings")


def evaluate_model(model, data_loader, criterion, device, print_loss=True):
    model.eval()
    overall_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Dynamically adjust input shape based on model type
            if isinstance(model, CNNRULRegression) or isinstance(model, CNNRULClassifier):
                inputs = inputs.permute(0, 2, 1)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.long()) # had to change from targets to targets.long so it would work for 1d CNN
            overall_loss += loss.item()

    avg_loss = overall_loss / len(data_loader)

    if print_loss:
        print(f"Test MSE: {avg_loss:.4f}")

    return avg_loss


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=5, track_test_loss=True):
    model.to(device)
    history = {'train_loss': [], 'test_loss': []}
    early_stopper = EarlyStopper(patience=3, min_delta=.01)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Adjust shape based on model type
            if isinstance(model, CNNRULRegression) or isinstance(model, CNNRULClassifier):  # 1D CNN
                inputs = inputs.permute(0, 2, 1)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.long()) # had to change from targets to targets.long so it would work for 1d CNN
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        if track_test_loss:
            avg_test_loss = evaluate_model(model, test_loader, criterion, device, print_loss=False)
            history['test_loss'].append(avg_test_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

            if early_stopper.early_stop(avg_test_loss):
                break
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

            if early_stopper.early_stop(avg_train_loss):
                break
        
    # if hasattr(model, 'feature_extractor'):
    #     log_embeddings(model, train_loader, device)

    return history


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


def plot_loss(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs. Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rul_predictions(model, test_loader, device):
    """
    Plots predicted RUL vs. actual RUL for the test set.

    Parameters:
        model (torch.nn.Module): Trained model for RUL prediction.
        test_loader (DataLoader): DataLoader containing test data.
        device (str): 'cuda' or 'cpu' based on availability.
    """
    predicted_rul, actual_rul = model.get_predict_and_true(test_loader, device)
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_rul, predicted_rul, alpha=0.5, label="Predicted vs Actual", color="blue")
    plt.plot([min(actual_rul), max(actual_rul)], [min(actual_rul), max(actual_rul)], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs. Actual RUL")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_accuracy(predictions, targets):
    correct_predictions = np.sum(predictions == targets)
    total_predictions = len(targets)
    return correct_predictions / total_predictions


def plot_confusion_matrix(y_true, y_pred):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
