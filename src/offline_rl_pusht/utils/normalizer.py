from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float32)
        return cls(
            mean=x.mean(axis=0).astype(np.float32),
            std=x.std(axis=0).astype(np.float32),
        )

    def transform_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (self.std + self.eps)

    def inverse_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x * (self.std + self.eps) + self.mean

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / (std + self.eps)

    def inverse_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * (std + self.eps) + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Standardizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            eps=float(state.get("eps", 1e-6)),
        )
