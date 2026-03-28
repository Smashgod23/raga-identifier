import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import os

# Load data — merge original CompMusic dataset with YouTube-sourced samples
X = np.load("data/X.npy")
y = np.load("data/y.npy")
with open("data/classes.json") as f:
    classes = json.load(f)

print(f"CompMusic: {len(X)} samples")

if os.path.exists("data/X_yt.npy"):
    X_yt = np.load("data/X_yt.npy")
    y_yt = np.load("data/y_yt.npy")
    X = np.concatenate([X, X_yt])
    y = np.concatenate([y, y_yt])
    print(f"YouTube:   {len(X_yt)} samples")

print(f"Total:     {len(X)} samples, {len(classes)} ragas")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# --- PyTorch training (for experimentation / best accuracy) ---

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)

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

model = RagaNet(input_dim=360, num_classes=len(classes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

best_acc = 0

for epoch in range(200):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

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

print(f"\nPyTorch training complete. Best accuracy: {best_acc:.1f}%")

# --- sklearn MLP for deployment (smaller Docker image, no PyTorch needed) ---

print("\nTraining sklearn MLPClassifier for deployment...")
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    alpha=0.01,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
    learning_rate='adaptive',
    learning_rate_init=0.001,
)
clf.fit(X, y)

with open("models/raga_sklearn.pkl", "wb") as f:
    pickle.dump(clf, f)

print(f"sklearn train accuracy: {clf.score(X, y)*100:.1f}%")
print(f"sklearn validation score: {clf.best_validation_score_*100:.1f}%")
print("Saved models/raga_sklearn.pkl")
