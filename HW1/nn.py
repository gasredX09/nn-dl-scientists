"""
You will need to implement a single layer neural network from scratch.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """

    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()
        self.x = None

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        # raise NotImplementedError()
        self.mask = x > 0
        return torch.where(self.mask, x, torch.zeros_like(x))

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        # raise NotImplementedError()
        return grad_wrt_out * self.mask.to(grad_wrt_out.dtype)

    def zerograd(self):
        return


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 * torch.rand(
            (outdim, indim), dtype=torch.float64, device=device
        )
        self.bias = 0.01 * torch.rand((outdim, 1), dtype=torch.float64, device=device)
        self.lr = lr

        # Placeholders for cached inputs and gradients
        self.x = None
        self.grad_w = torch.zeros_like(self.weights)
        self.grad_b = torch.zeros_like(self.bias)

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        # raise NotImplementedError()
        self.x = x
        return self.weights @ x + self.bias

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        # compute grad_wrt_weights
        # dL/dW = (dL/dy) @ x^T
        self.grad_w = grad_wrt_out @ self.x.T

        # compute grad_wrt_bias
        # dL/db = sum over batch (keep column shape)
        self.grad_b = grad_wrt_out.sum(dim=1, keepdim=True)

        # compute & return grad_wrt_input
        # dL/dx = W^T @ (dL/dy)
        grad_wrt_input = self.weights.T @ grad_wrt_out
        return grad_wrt_input

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        self.weights -= self.lr * self.grad_w
        self.bias -= self.lr * self.grad_b

    def zerograd(self):
        self.grad_w.zero_()
        self.grad_b.zero_()


class SoftmaxCrossEntropyLoss(object):
    def __init__(self):
        """
        Initialize any parameters
        """
        self.logits = None
        self.labels = None
        self.batch_size = None
        self.accu = None

    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        self.labels = labels
        self.batch_size = logits.shape[1]

        # stable softmax
        shifted = logits - logits.max(dim=0, keepdim=True).values
        exp_scores = torch.exp(shifted)
        self.probs = exp_scores / exp_scores.sum(dim=0, keepdim=True)

        # cross entropy loss
        eps = 1e-12  # to avoid log(0)
        logp = torch.log(self.probs + eps)
        loss_per_sample = -(labels * logp).sum(dim=0)  # shape (batch_size,)
        loss = loss_per_sample.mean()  # scalar

        # accuracy cache
        pred = torch.argmax(self.probs, dim=0)
        true = torch.argmax(labels, dim=0)
        self.accu = (pred == true).to(torch.float64).mean().item()

        return loss

    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        return (self.probs - self.labels) / self.batch_size

    def getAccu(self):
        """
        return accuracy here
        """
        return self.accu


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""

    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        self.fc1 = LinearMap(indim, hidden_layer, lr)
        self.relu = ReLU()
        self.fc2 = LinearMap(hidden_layer, outdim, lr)

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        out = self.fc1.forward(x)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        return out

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        grad = self.fc2.backward(grad_wrt_out)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

    def step(self):
        """update model parameters"""
        self.fc1.step()
        self.fc2.step()

    def zerograd(self):
        self.fc1.zerograd()
        self.fc2.zerograd()


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


def labels2onehot(labels: np.ndarray):
    return np.array([[i == lab for i in range(2)] for lab in labels]).astype(int)


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
    Ytest = pd.read_csv("./data/y_test.csv")
    Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # construct the model
    criterion = SoftmaxCrossEntropyLoss()
    model = SingleLayerMLP(indim, outdim, hidden_dim, lr)

    train_losses = []
    train_accus = []
    test_losses = []
    test_accus = []

    for epoch in range(epochs):
        model.zerograd()
        epoch_loss = 0.0
        epoch_accu = 0.0
        num_batches = 0

        # TRAINING LOOP
        for x_batch, y_batch in train_loader:
            # prepare inputs
            x = x_batch.T.to(device=device, dtype=torch.float64)
            y_onehot = labels2onehot(y_batch.numpy())
            y = torch.tensor(y_onehot.T, dtype=torch.float64, device=device)

            # forward
            logits = model.forward(x)
            loss = criterion.forward(logits, y)

            # backward
            grad_logits = criterion.backward()
            model.backward(grad_logits)

            # update
            model.step()
            model.zerograd()

            epoch_loss += loss.item()
            epoch_accu += criterion.getAccu()
            num_batches += 1

        train_losses.append(epoch_loss / num_batches)
        train_accus.append(epoch_accu / num_batches)

        # TESTING LOOP
        with torch.no_grad():
            test_loss = 0.0
            test_accu = 0.0
            num_batches = 0

            for x_batch, y_batch in test_loader:
                x = x_batch.T.to(device=device, dtype=torch.float64)
                y_onehot = labels2onehot(y_batch.numpy())
                y = torch.tensor(y_onehot.T, dtype=torch.float64, device=device)

                logits = model.forward(x)
                loss = criterion.forward(logits, y)

                test_loss += loss.item()
                test_accu += criterion.getAccu()
                num_batches += 1

            test_losses.append(test_loss / num_batches)
            test_accus.append(test_accu / num_batches)

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
    plt.savefig("loss_accuracy_curves_nn.png", dpi=300)
    plt.show()
