import torch


def project_simplex(v: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    n = v.shape[0]
    u = v.sort(descending=True).values
    cssv = u.cumsum(dim=0)
    rho = ((u * torch.arange(1, n + 1, device=v.device) > cssv - s).sum() - 1).clamp(min=0)
    theta = (cssv[rho] - s) / (rho + 1)
    w = (v - theta).clamp(min=0)
    return w
