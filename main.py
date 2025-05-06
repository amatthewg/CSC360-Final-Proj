import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# Set the path to the various letters (from https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)
# Change this if your path is different
TRAIN_DIR = "asl_alphabet_train"
TEST_DIR = "asl_alphabet_test"
DATA_CLASSES = ("A", "B", "I", "L", "nothing", "R", "U", "V", "W")

# Set ratio of data
TRAIN_DATA_RATIO = 0.8 # Ratio of the *selected 1/3* for training
VAL_DATA_RATIO = 0.2   # Ratio of the *selected 1/3* for validation
TRAIN_VAL_POOL_RATIO = 1/3 # Ratio of the total data to use for train/val pool

LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LetterDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {class_name: i for i, class_name in enumerate(DATA_CLASSES)}
        self.images = []

        if root_dir == TRAIN_DIR:
            all_train_images = []
            for class_name in DATA_CLASSES:
                class_path = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_path):
                    all_train_images.append((os.path.join(class_path, img_name), self.class_to_idx[class_name]))

            random.shuffle(all_train_images)
            total_train_samples = len(all_train_images)
            train_val_pool_size = int(total_train_samples * TRAIN_VAL_POOL_RATIO)
            train_val_pool = all_train_images[:train_val_pool_size]

            if subset == 'train':
                train_size = int(len(train_val_pool) * TRAIN_DATA_RATIO)
                self.images = train_val_pool[:train_size]
            elif subset == 'val':
                train_size = int(len(train_val_pool) * TRAIN_DATA_RATIO)
                self.images = train_val_pool[train_size:]
            elif subset is None:
                self.images = all_train_images

        elif root_dir == TEST_DIR:
            for img_name in os.listdir(root_dir):
                prefix = img_name.split('_')[0]
                if prefix in DATA_CLASSES:
                    self.images.append((os.path.join(root_dir, img_name), self.class_to_idx[prefix]))
            # Test subset should encompass all test images
            if subset == 'test':  # Already loaded all test images
                pass
            elif subset is not None:
                print(f"Warning: Subset '{subset}' is not applicable for the test directory.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class HandSignalCNN(nn.Module):
    def __init__(self, num_classes=len(DATA_CLASSES)):
        super(HandSignalCNN, self).__init__()
        # Input: 200x200 RGB image (3 channels)

        # Conv block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Outputs 16x100x100

        # Conv block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Outputs 32x50x50

        # Conv block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # Output: 64x25x25

        # Fully Connected Layers
        # Corrected in_features from 64 * 25 * 25 to 64 * 24 * 24
        self.fc1 = nn.Linear(in_features=64 * 24 * 24, out_features=256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 64 * 24 * 24)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)  # Output raw logits, CrossEntropyLoss will handle softmax

        return x


if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5283, 0.5081, 0.5241], std=[0.2272, 0.2542, 0.2600])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5260, 0.5060, 0.5223], std=[0.2282, 0.2546, 0.2604])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4725, 0.4668, 0.4785], std=[0.2065, 0.2454, 0.2585])
    ])

    print(f"Loading training data from: {TRAIN_DIR}")
    train_dataset = LetterDataset(TRAIN_DIR, transform=train_transform, subset='train')
    val_dataset = LetterDataset(TRAIN_DIR, transform=val_transform, subset='val')

    print(f"Loading testing data from: {TEST_DIR}")
    test_dataset = LetterDataset(TEST_DIR, transform=test_transform, subset='test')

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = HandSignalCNN().to(device)
    print("\nModel Architecture:")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training...")
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            #if (i + 1) % 100 == 0:  # Print progress every 100 batches
                #print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = epoch_train_loss / total_train if total_train > 0 else 0
        train_acc = correct_train / total_train if total_train > 0 else 0
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation loop
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images.nelement() == 0:
                    continue
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                epoch_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        if total_val > 0:
            avg_val_loss = epoch_val_loss / total_val
            val_acc = correct_val / total_val
            print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
        else:
            print("Validation set was empty or all images failed to load.")

    print("\nTraining Finished")

    # Final testing
    model.eval()
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            if images.nelement() == 0:
                continue
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if total_test > 0:
        test_accuracy = 100 * correct_test / total_test
        print(f'Accuracy of the model on the {total_test} test images: {test_accuracy:.2f} %')

    print("Train losses:")
    print(train_losses)
    print("Train accuracies:")
    print(train_accuracies)
    print("Val losses:")
    print(val_losses)
    print("Val accuracies:")
    print(val_accuracies)
    if len(train_losses) > 0 and len(val_losses) > 0:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
