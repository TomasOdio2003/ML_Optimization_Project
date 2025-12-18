from __future__ import annotations
import os
import time
import pandas as pd
import torch

class CSVLogger:
    def __init__(self, out_csv: str):
        self.out_csv = out_csv
        self.rows = []
        self.t0 = time.time()
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    def log(self, k: int, f: torch.Tensor, g: torch.Tensor, alpha, phase: str = "main", **extras):
        def coerce(v):
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return float(v)
                return float(torch.linalg.norm(v))  # fallback if someone logs a vector
            if v is None:
                return None
            if isinstance(v, (int, float, str, bool)):
                return v
            try:
                return float(v)
            except Exception:
                return str(v)

        row = {
            "iter": int(k),
            "phase": phase,
            "loss": float(f),
            "grad_norm": float(torch.linalg.norm(g)),
            "alpha": None if alpha is None else float(alpha),
            "elapsed_sec": time.time() - self.t0,
        }
        row.update({kk: coerce(vv) for kk, vv in extras.items()})
        self.rows.append(row)




    def close(self):
        pd.DataFrame(self.rows).to_csv(self.out_csv, index=False)
