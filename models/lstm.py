import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)
    
def evaluate_model(model, val_loader):
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            all_preds += (outputs.cpu().numpy() > 0.5).astype(int).flatten().tolist()
            all_labels += labels.cpu().numpy().flatten().tolist()

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            all_preds += (outputs.detach().cpu().numpy() > 0.5).astype(int).flatten().tolist()
            all_labels += labels.cpu().numpy().flatten().tolist()

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {np.mean(train_losses):.4f} - Accuracy: {acc:.4f}")

        evaluate_model(model, val_loader)

