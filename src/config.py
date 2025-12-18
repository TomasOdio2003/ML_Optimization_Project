from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cpu"

    # Data
    dataset: str = "sine"   # "sine" or "ill_cond"
    m: int = 1000           # number of samples (full batch)
    d: int = 1              # input dimension
    noise_std: float = 0.05

    # Model (fixed per your choice)
    width: int = 16

    # Optimization
    max_iters: int = 200
    tol_grad: float = 1e-6
    alpha0: float = 1.0
    use_line_search: bool = True

    # Preconditioning
    damping: float = 1e-4          # added to diag (stability)
    precond_refresh: int = 10      # recompute diag GN every N iters
    
    # L-BFGS
    lbfgs_history_size: int = 10
    lbfgs_build_iters: int = 80   # for pgd_lbfgs_precond: build H^{-1} near solution
    
    # Gauss–Newton + CG/PCG
    gn_outer_iters: int = 40
    gn_damping: float = 1e-3
    cg_tol: float = 1e-6
    cg_max_iters: int = 200
    gn_use_line_search: bool = True


