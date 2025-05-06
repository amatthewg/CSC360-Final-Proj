from main import LetterDataset, set_seed, TRAIN_DIR, TEST_DIR
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def mean_std(loader):
    imgs = torch.stack([img_t for img_t, _ in loader], dim=3)
    mean = imgs.view(3, -1).mean(dim=1)
    std = imgs.view(3, -1).std(dim=1)
    return mean, std


if __name__ == "__main__":
    set_seed()
    initial_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])
    train_dataset = LetterDataset(TRAIN_DIR, initial_transform, 'train')
    val_dataset = LetterDataset(TRAIN_DIR, initial_transform, 'val')
    test_dataset = LetterDataset(TEST_DIR, initial_transform, 'test')

    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset)

    train_mean, train_std = mean_std(train_loader)
    print(f"Training Dataset Mean: {train_mean}")
    print(f"Training Dataset Std: {train_std}")

    val_mean, val_std = mean_std(val_loader)
    print(f"val Dataset Mean: {val_mean}")
    print(f"val Dataset Std: {val_std}")

    test_mean, test_std = mean_std(test_loader)
    print(f"Testing Dataset Mean: {test_mean}")
    print(f"Testing Dataset Std: {test_std}")