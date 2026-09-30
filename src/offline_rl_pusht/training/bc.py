from __future__ import annotations
import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.MSELoss(reduction="sum")
    total_loss, n = 0.0, 0

    for states, actions in loader:
        states, actions = states.to(device), actions.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(states)
        loss = criterion(pred, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += states.shape[0]

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss, n = 0.0, 0

    for states, actions in loader:
        states, actions = states.to(device), actions.to(device)
        total_loss += criterion(model(states), actions).item()
        n += states.shape[0]

    return total_loss / max(n, 1)
