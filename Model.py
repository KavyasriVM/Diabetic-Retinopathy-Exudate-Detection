import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# 2. Dataset Class
class ExudateDataset(Dataset):
    def _init_(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (384, 384))
        image = np.stack([image]*3, axis=-1)  # Convert to 3-channel
       
        if self.transform:
            image = self.transform(image)
           
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)

# 3. Classification Model
class FPN_EfficientNetB0_Classifier(nn.Module):
    def _init_(self, num_classes=1):
        super()._init_()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
       
        # Corrected FPN channels based on EfficientNet-B0's actual outputs
        self.lateral4 = nn.Conv2d(320, 128, 1)  # reduction_5 (320 channels)
        self.lateral3 = nn.Conv2d(112, 128, 1)  # reduction_4 (112 channels)
        self.lateral2 = nn.Conv2d(40, 128, 1)   # reduction_3 (40 channels)
        self.lateral1 = nn.Conv2d(24, 128, 1)   # reduction_2 (24 channels)
       
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Get correct endpoints from EfficientNet-B0
        endpoints = self.base.extract_endpoints(x)
        c1 = endpoints['reduction_2']  # 24 channels
        c2 = endpoints['reduction_3']  # 40 channels
        c3 = endpoints['reduction_4']  # 112 channels
        c4 = endpoints['reduction_5']  # 320 channels
       
        # Feature Pyramid Network
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear')
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear')
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear')
       
        # Classification
        x = self.gap(p1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 4. Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=50, patience=5, device='cuda'):
    model.to(device)
    scaler = GradScaler()
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    accumulation_steps = 4  # Gradient accumulation
   
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
       
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
           
            with autocast(device_type='cuda'):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels) / accumulation_steps
               
            scaler.scale(loss).backward()
           
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
           
            train_loss += loss.item() * inputs.size(0)
       
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
       
        # Update metrics
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
       
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
       
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
       
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f}')
   
    return model, history

# 5. Main Execution
def main():
    # Dataset setup
    DATA_PATH = "/kaggle/input/preprocessed/Preprocessed_APTOS_background"
   
    def prepare_data(data_path, max_per_class=4619):
        valid_exts = ('.png', '.jpg', '.jpeg')
        class0 = sorted([os.path.join(data_path, '0', f)
                        for f in os.listdir(os.path.join(data_path, '0'))
                        if f.lower().endswith(valid_exts)])[:max_per_class]
        class3 = sorted([os.path.join(data_path, '3', f)
                        for f in os.listdir(os.path.join(data_path, '3'))
                        if f.lower().endswith(valid_exts)])[:max_per_class]
        return class0 + class3, [0]*len(class0) + [1]*len(class3)
   
    images, labels = prepare_data(DATA_PATH)
   
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
   
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
   
    # Create datasets
    BATCH_SIZE = 8
    train_dataset = ExudateDataset(X_train, y_train, train_transform)
    val_dataset = ExudateDataset(X_val, y_val, val_test_transform)
    test_dataset = ExudateDataset(X_test, y_test, val_test_transform)
   
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
   
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPN_EfficientNetB0_Classifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    torch.cuda.empty_cache()

    # Train
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer)
   
    # Evaluate
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            test_correct += (preds == labels).sum().item()
   
    print(f'\nFinal Test Accuracy: {test_correct/len(test_dataset):.4f}')
   
    # Plot results
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.legend()
    plt.show()

if _name_ == "_main_":
    main()
