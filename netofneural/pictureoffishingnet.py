# pyright: basic
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class FaceDataset(Dataset):
    def __init__(self, imgs, pts, augment=False):
        self.imgs = imgs
        self.pts = pts / 256.0  # normalise to [0,1]
        self.augment = augment
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.resize(self.imgs[idx], (96, 96))
        pts = self.pts[idx].flatten().astype(np.float32)
        if self.augment:
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((48, 48), angle, 1.0)
            img = cv2.warpAffine(img, M, (96, 96))
            ones = np.ones((5, 1))
            pts_2d = pts.reshape(5, 2) * 96
            pts_2d = (M @ np.hstack([pts_2d, ones]).T).T
            pts = (pts_2d / 96).flatten().astype(np.float32)
        return self.transform(img), torch.tensor(pts)


class FaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def train(img_train, pts_train, img_val, pts_val, epochs):
    train_dl = DataLoader(
        FaceDataset(img_train, pts_train, augment=True), batch_size=32, shuffle=True
    )
    val_dl = DataLoader(FaceDataset(img_val, pts_val), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for imgs, pts in train_dl:
            imgs, pts = imgs.to(device), pts.to(device)
            opt.zero_grad()
            loss_fn(model(imgs), pts).backward()
            opt.step()

        model.eval()
        val_loss = sum(
            loss_fn(model(imgs.to(device)), pts.to(device)).item()
            for imgs, pts in val_dl
        ) / len(val_dl)
        print(f"epoch {epoch+1:3d} | val_loss {val_loss:.6f}")

    torch.save(model.state_dict(), "face_cnn.pth")
    return model
