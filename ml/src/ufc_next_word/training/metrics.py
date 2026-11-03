"""Loss and top-k metrics shared by validation and final testing."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    top1_accuracy: float
    top3_accuracy: float
    examples: int

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "examples": self.examples,
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    top1_correct = 0
    top3_correct = 0

    for contexts, targets in loader:
        contexts = contexts.to(device)
        targets = targets.to(device)
        logits = model(contexts)
        batch_size = targets.size(0)
        total_loss += criterion(logits, targets).item() * batch_size
        total_examples += batch_size

        top_predictions = logits.topk(k=min(3, logits.size(1)), dim=1).indices
        top1_correct += (top_predictions[:, 0] == targets).sum().item()
        top3_correct += (
            (top_predictions == targets.unsqueeze(1)).any(dim=1).sum().item()
        )

    denominator = max(total_examples, 1)
    return EvaluationResult(
        loss=total_loss / denominator,
        top1_accuracy=top1_correct / denominator,
        top3_accuracy=top3_correct / denominator,
        examples=total_examples,
    )

