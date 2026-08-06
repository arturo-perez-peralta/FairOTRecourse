import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class TransportResult:
    plan: Optional[torch.Tensor] = None
    f: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    converged: bool = False
    n_iter: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseSolver(ABC):
    def __init__(self, epsilon: float = 0.1, max_iter: int = 1000, tol: float = 1e-9):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol

    @abstractmethod
    def solve(
        self,
        mu0: torch.Tensor,
        mu1: torch.Tensor,
        cost_matrix: torch.Tensor,
        groups: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> TransportResult:
        ...
