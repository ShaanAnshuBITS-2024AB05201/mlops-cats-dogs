import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import numpy as np
import os

DATA_DIR = 'data/processed'
MODEL_PATH = 'models/model.pt'
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def build_model():
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    return model

def train():
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model  = build_model().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mlflow.set_experiment('cats-vs-dogs')
    with mlflow.start_run():
        mlflow.log_params({'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR, 'model': 'mobilenet_v2'})

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_dl)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = (torch.sigmoid(model(imgs)).squeeze() > 0.5).long()
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)

            val_acc = correct / total
            mlflow.log_metrics({'train_loss': avg_loss, 'val_accuracy': val_acc}, step=epoch)
            print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}')

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs = imgs.to(device)
                preds = (torch.sigmoid(model(imgs)).squeeze() > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        test_acc = np.trace(cm) / np.sum(cm)
        mlflow.log_metric('test_accuracy', test_acc)
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Confusion Matrix:\n{cm}')

        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        mlflow.pytorch.log_model(model, 'model')
        print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    train()
