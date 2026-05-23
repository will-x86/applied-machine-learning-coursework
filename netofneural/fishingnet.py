# pyright:basic
import numpy as np
import torch
import torch.nn as nn


class SentimentNLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # single logit -> BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def predict_mlp(model: SentimentNLP, X: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return (logits.sigmoid() >= 0.5).cpu().numpy().astype(int)


def train_mlp(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    dropout: float,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 15,
    batch_size: int = 128,
) -> SentimentNLP:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    model = SentimentNLP(X_train.shape[1], hidden_dim, dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(xb)
        print(f"  epoch {epoch+1:>2}/{epochs}  loss={total_loss/len(X_train):.4f}")

    model.eval()
    return model
