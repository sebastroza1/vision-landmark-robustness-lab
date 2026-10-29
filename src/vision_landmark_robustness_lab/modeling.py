from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LandmarkMLP(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def predict_mlp(model: LandmarkMLP, scaler: StandardScaler, features: np.ndarray) -> np.ndarray:
    model.eval()
    transformed = scaler.transform(features).astype(np.float32)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(transformed)))
    return probabilities.numpy()


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    epochs: int,
) -> tuple[LandmarkMLP, StandardScaler, list[dict[str, float]]]:
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train_x)
    scaled_train = scaler.transform(train_x).astype(np.float32)
    model = LandmarkMLP(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.from_numpy(scaled_train), torch.from_numpy(train_y.astype(np.float32))
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history = []
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
        predictions = predict_mlp(model, scaler, validation_x) >= 0.5
        validation_f1 = float(f1_score(validation_y, predictions, average="macro"))
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": total_loss / len(dataset),
                "validation_macro_f1": validation_f1,
            }
        )
        if validation_f1 > best_f1 + 1e-5:
            best_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 8:
                break
    model.load_state_dict(best_state)
    return model, scaler, history


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }

