import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from tqdm import tqdm


class DS(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = int(self.labels[idx])
        image = self.transform(image)
        return image, label


class Test_DS(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        image = self.transform(image)
        return image


class History:
    def __init__(self, device):
        self.history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        self.device = device
        self.verbose = None
        self.pbar = None
        self.epochs = None
        self.epoch = None

    def training_loop(self, dataloader, model, loss_fn, optimizer, show_log=True):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        loss, accuracy = 0.0, 0.0
        total = math.ceil(size / dataloader.batch_size)
        self.pbar.reset(total=total)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            # Compute prediction and loss
            pred = model(X)

            # MSELoss expects float
            y = y.float()

            # kill broadcasting warning
            y = y.unsqueeze(-1)

            batch_loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            batch_loss = batch_loss.item()
            loss += batch_loss
            # batch_accuracy = torch.mean((pred.argmax(1) == y).type(torch.float)).item()
            batch_accuracy = torch.mean(((pred >= 0.5) == y).type(torch.float)).item()
            accuracy += batch_accuracy

            self.pbar.update()
            if self.verbose == 0:
                self.pbar.set_postfix(
                    Epoch=f"{self.epoch+1}/{self.epochs}",
                    accuracy=f"{batch_accuracy:.4f}",
                    loss=f"{batch_loss:.4f}",
                )
            elif self.verbose == 1:
                self.pbar.set_postfix(
                    accuracy=f"{batch_accuracy:.4f}",
                    loss=f"{batch_loss:.4f}",
                )
        loss /= num_batches
        accuracy /= num_batches
        return accuracy, loss

    def validation_loop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        loss, accuracy = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                # kill broadcasting warning
                y = y.unsqueeze(-1)

                loss += loss_fn(pred, y).item()
                # accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
                accuracy += ((pred >= 0.5) == y).sum().item()

        loss /= num_batches
        accuracy /= size
        training_accuracy = self.history["accuracy"][-1]
        training_loss = self.history["loss"][-1]
        if self.verbose == 0:
            self.pbar.set_postfix(
                Epoch=f"{self.epoch+1}/{self.epochs}",
                accuracy=f"{training_accuracy:.4f}",
                loss=f"{training_loss:.4f}",
                val_accuracy=f"{accuracy:.4f}",
                val_loss=f"{loss:.4f}",
            )
        elif self.verbose == 1:
            self.pbar.set_postfix(
                accuracy=f"{training_accuracy:.4f}",
                loss=f"{training_loss:.4f}",
                val_accuracy=f"{accuracy:.4f}",
                val_loss=f"{loss:.4f}",
            )
            self.pbar.close()
        return accuracy, loss

    def fit_model(
        self,
        model,
        training_dataloader,
        loss_fn,
        optimizer,
        epochs,
        verbose=1,
        validation_dataloader=None,
    ):
        if verbose == 0:
            self.pbar = tqdm()
        self.verbose = verbose
        self.epochs = epochs
        for epoch in range(epochs):
            self.epoch = epoch
            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs}")
                self.pbar = tqdm()
            accuracy, loss = self.training_loop(
                training_dataloader, model, loss_fn, optimizer
            )
            self.history["accuracy"].append(accuracy)
            self.history["loss"].append(loss)
            if validation_dataloader is not None:
                val_accuracy, val_loss = self.validation_loop(
                    validation_dataloader, model, loss_fn
                )
                self.history["val_accuracy"].append(val_accuracy)
                self.history["val_loss"].append(val_loss)
            elif validation_dataloader is None and self.verbose == 1:
                self.pbar.close()
        if verbose == 0:
            self.pbar.close()

    def predict(self, dataloader, model):
        preds = []
        model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                # pred = torch.ravel(pred)
                preds += pred.cpu().numpy().tolist()
        return preds

    def predict_test(self, dataloader, model):
        preds = []
        size = len(dataloader.dataset)
        total = math.ceil(size / dataloader.batch_size)
        pbar = tqdm(total=total)
        model.eval()
        with torch.no_grad():
            for X in dataloader:
                X = X.to(self.device)
                pred = model(X)
                # pred = torch.ravel(pred)
                preds += pred.cpu().numpy().tolist()
                pbar.update(1)
        pbar.close()
        return preds
