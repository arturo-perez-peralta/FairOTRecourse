import torch
import pytest
from fairopt.utils.simplex import project_simplex


class TestSimplexProjection:
    def test_projection_on_simplex(self):
        v = torch.tensor([0.5, 0.5, 0.5])
        w = project_simplex(v, s=1.0)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
        assert (w >= 0).all()

    def test_already_on_simplex(self):
        v = torch.tensor([0.3, 0.2, 0.5])
        w = project_simplex(v, s=1.0)
        assert torch.allclose(v, w, atol=1e-6)

    def test_negative_values(self):
        v = torch.tensor([-0.5, 0.8, 0.9])
        w = project_simplex(v, s=1.0)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
        assert (w >= 0).all()

    def test_sum_s(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        s = 5.0
        w = project_simplex(v, s=s)
        assert torch.allclose(w.sum(), torch.tensor(s), atol=1e-6)

    def test_all_negative(self):
        v = torch.tensor([-1.0, -2.0, -3.0])
        w = project_simplex(v, s=1.0)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
        assert (w >= 0).all()
