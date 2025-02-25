import torch
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
data_structures_path = os.path.join(parent_dir, "data_structures")
sys.path.append(data_structures_path)
from data_structures import EarlyStopper
from .regression import CNNRULRegression
from .classification import CNNRULClassifier


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
            loss = criterion(outputs, targets)
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
            loss = criterion(outputs, targets)
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

    return history


def plot_loss(history, save_as=None):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs. Test Loss')
    plt.legend()
    plt.grid(True)
    if save_as:
        plt.savefig(save_as)
    plt.show()



def plot_rul_predictions(model, test_loader, device, save_as=None, aggregate=False, shaded_region=False):
    """
    Plots predicted RUL vs. actual RUL for the test set.

    Parameters:
        model (torch.nn.Module): Trained model for RUL prediction.
        test_loader (DataLoader): DataLoader containing test data.
        device (str): 'cuda' or 'cpu' based on availability.
        save_as (str, optional): Path to save the plot. If None, the plot will not be saved.
        aggregate (bool, optional): If True, aggregates predictions for unique actual RUL values.
        shaded_region (bool, optional): If True, shades the region between min and max predictions for each unique
            actual RUL value.
    """
    predicted_rul, actual_rul = model.get_predict_and_true(test_loader, device)
    plt.figure(figsize=(8, 6), facecolor='#384957')

    if aggregate or shaded_region:
        # Convert to Pandas DataFrame for efficient grouping
        df = pd.DataFrame({"actual_rul": actual_rul, "predicted_rul": predicted_rul})
        grouped = df.groupby("actual_rul")["predicted_rul"]

        # Compute mean, min, and max for each actual RUL value
        mean_pred = grouped.mean()
        min_pred = grouped.min()
        max_pred = grouped.max()

        # Sort by actual RUL for proper plotting
        sorted_actual = mean_pred.index
        sorted_mean_pred = mean_pred.values
        sorted_min_pred = min_pred.values
        sorted_max_pred = max_pred.values

        if shaded_region:
            plt.fill_between(sorted_actual, sorted_min_pred, sorted_max_pred, color="#FF6B65", alpha=0.15,
                             label="Prediction Range")

        if aggregate:
            plt.plot(sorted_actual, sorted_mean_pred, color="#FF6B65", label="Mean Prediction")
    else:
        plt.scatter(actual_rul, predicted_rul, alpha=0.5, label="Predicted vs Actual", color="#FF6B65", s=5)
    #plt.scatter(actual_rul, predicted_rul, alpha=0.5, label="Predicted vs Actual", color="blue", s=5)
    plt.plot([min(actual_rul), max(actual_rul)], [min(actual_rul), max(actual_rul)], 'r--', label="Perfect Prediction", color="#384957")
    plt.xlabel("Actual RUL", color="white")
    plt.ylabel("Predicted RUL", color="white")
    plt.title("Predicted vs. Actual RUL", color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.legend()
    plt.grid(True)
    if save_as:
        plt.savefig(save_as)
    plt.show()

# main line ff6b65
# shaded region 384957
# perfect pred line c2c2c2


def calculate_accuracy(predictions, targets):
    correct_predictions = np.sum(predictions == targets)
    total_predictions = len(targets)
    return correct_predictions / total_predictions


def plot_confusion_matrix(y_true, y_pred, save_as=None):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    if save_as:
        plt.savefig(save_as)
    plt.show()
