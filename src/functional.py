from __future__ import annotations
from collections import OrderedDict
import torch

try:
    from torch.func import functional_call
except Exception as e:
    raise RuntimeError("This project requires torch>=2.0 for torch.func.functional_call") from e

def params_to_template(model) -> OrderedDict[str, torch.Tensor]:
    # Template only stores shapes; values don't matter.
    return OrderedDict((k, v.detach()) for k, v in model.named_parameters())

def flatten_params(params: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params.values()])

def unflatten_vec(vec: torch.Tensor, template: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Turn a flat vector into a param dict with views into vec (keeps grad path!)."""
    out = OrderedDict()
    i = 0
    for name, t in template.items():
        n = t.numel()
        out[name] = vec[i:i+n].view_as(t)
        i += n
    return out

def residuals_from_theta_vec(model, template, theta_vec: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    params = unflatten_vec(theta_vec, template)
    yhat = functional_call(model, params, (X,))
    r = (yhat - y).reshape(-1)  # (m,)
    return r

def loss_and_grad(model, template, theta_vec: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
    theta_vec = theta_vec.detach().requires_grad_(True)
    r = residuals_from_theta_vec(model, template, theta_vec, X, y)
    loss = 0.5 * (r @ r)
    (g,) = torch.autograd.grad(loss, theta_vec, create_graph=False)
    return loss.detach(), g.detach()
