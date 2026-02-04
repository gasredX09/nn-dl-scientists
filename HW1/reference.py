"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modules in this code.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""

    def __init__(self, indim, outdim, hidden_layer=100):
        super(SingleLayerMLP, self).__init__()
        self.fc1 = nn.Linear(indim, hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, outdim)

    def forward(self, x):
        """
        x shape (batch_size, indim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length


def validate(loader):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x = x_batch.to(device=device, dtype=torch.float32)
            y = y_batch.to(device=device, dtype=torch.long)

            logits = model(x)
            loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)

    mean_loss = total_loss / total_count
    mean_acc = total_correct / total_count
    return mean_loss, mean_acc


if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """

    indim = 60
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 500

    # dataset
    Xtrain = pd.read_csv("./data/X_train.csv")
    Ytrain = pd.read_csv("./data/y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = pd.DataFrame(
        scaler.fit_transform(Xtrain), columns=Xtrain.columns
    ).to_numpy()
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv("./data/X_test.csv")
    Ytest = pd.read_csv("./data/y_test.csv").to_numpy()
    Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # construct the model
    model = SingleLayerMLP(indim, outdim, hidden_dim).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses, train_accus = [], []
    test_losses, test_accus = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x_batch, y_batch in train_loader:
            x = x_batch.to(device=device, dtype=torch.float32)
            y = y_batch.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)

        train_losses.append(total_loss / total_count)
        train_accus.append(total_correct / total_count)

        test_loss, test_acc = validate(test_loader)
        test_losses.append(test_loss)
        test_accus.append(test_acc)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"Train Loss {train_losses[-1]:.4f}, "
                f"Train Acc {train_accus[-1]:.4f}, "
                f"Test Acc {test_accus[-1]:.4f}"
            )

    # PLOTTING
    epochs_range = range(epochs)

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].plot(epochs_range, train_losses, label="Train Loss")
    axs[0].plot(epochs_range, test_losses, label="Test Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss vs Epoch")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs_range, train_accus, label="Train Accuracy")
    axs[1].plot(epochs_range, test_accus, label="Test Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy vs Epoch")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("loss_accuracy_curves_reference.png", dpi=300)
    plt.show()
