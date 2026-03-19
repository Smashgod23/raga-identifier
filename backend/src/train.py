import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load data
X = np.load("data/X.npy")
y = np.load("data/y.npy")
with open("data/classes.json") as f:
    classes = json.load(f)

print(f"Loaded {len(X)} samples, {len(classes)} ragas")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split 80/20 train/test
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)

# Model
class RagaNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = RagaNet(input_dim=240, num_classes=len(classes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# Train
best_acc = 0
os.makedirs("models", exist_ok=True)

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    acc = correct / total * 100

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/raga_model_best.pt")

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.1f}% | Best: {best_acc:.1f}%")

print(f"\nTraining complete. Best accuracy: {best_acc:.1f}%")
print("Model saved to models/raga_model_best.pt")