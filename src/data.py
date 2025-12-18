import torch

def make_sine_data(m: int, d: int, noise_std: float, device: str):
    # X in [-2, 2]^d
    X = -2.0 + 4.0 * torch.rand(m, d, device=device)
    x0 = X[:, [0]]  # (m,1) use first coordinate for the sine target
    y_clean = torch.sin(3.0 * x0) + 0.3 * torch.sin(9.0 * x0)
    y = y_clean + noise_std * torch.randn_like(y_clean)
    return X, y


def make_ill_conditioned_data(m: int, d: int, noise_std: float, device: str):
    # Features with wildly different scales -> conditioning pain
    X = torch.randn(m, d, device=device)
    scales = (10.0 ** torch.linspace(0, d - 1, d, device=device)).view(1, d)  # 1,10,100,...
    X = X * scales

    w = torch.randn(d, 1, device=device) / (d ** 0.5)
    y_clean = torch.tanh(X @ w)
    y = y_clean + noise_std * torch.randn_like(y_clean)
    return X, y

def make_dataset(name: str, m: int, d: int, noise_std: float, device: str):
    if name == "sine":
        return make_sine_data(m=m, d=d, noise_std=noise_std, device=device)
    if name == "ill_cond":
        return make_ill_conditioned_data(m=m, d=d, noise_std=noise_std, device=device)
    raise ValueError(f"Unknown dataset: {name}")
