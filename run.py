import argparse
import torch
from src.optimizers import lbfgs_minimize

from src.optimizers import lbfgs_d_minimize

from src.config import ExperimentConfig
from src.data import make_dataset
from src.model import MLP16
from src.functional import params_to_template, flatten_params, loss_and_grad
from src.preconditioners import DiagGNPreconditioner
from src.optimizers import gradient_descent, preconditioned_gd
from src.logging_utils import CSVLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="sine", choices=["sine", "ill_cond"])
    ap.add_argument("--method", type=str, default="gd",
                choices=[
  "gd", "pgd_diag_gn", "lbfgs", "lbfgs_d"
])
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0,
                help="Seed used ONLY for dataset generation (kept constant across runs).")
    args = ap.parse_args()

    cfg = ExperimentConfig(dataset=args.dataset, m=args.m, d=args.d, max_iters=args.iters, seed=args.seed)

    device = torch.device(cfg.device)

    # 1) FIXED dataset across runs (same X,y for all methods + init seeds)
    torch.manual_seed(args.data_seed)
    X, y = make_dataset(cfg.dataset, cfg.m, cfg.d, cfg.noise_std, device=str(device))

    # 2) VARY model initialization across runs (controlled by --seed)
    torch.manual_seed(cfg.seed + 12345)


    model = MLP16(d=cfg.d, width=cfg.width, activation="tanh").to(device)
    template = params_to_template(model)
    theta0 = flatten_params(template).to(device)  # start from the model's init params

    print("theta0 checksum:", float(theta0.sum()), float((theta0**2).sum()))
    print("X checksum:", float(X.sum()), float((X**2).sum()))
    print("y checksum:", float(y.sum()), float((y**2).sum()))

    def obj_fn(theta_vec):
        return loss_and_grad(model, template, theta_vec, X, y)

    logger = CSVLogger(out_csv=f"results/{cfg.dataset}_{args.method}_m{cfg.m}_d{cfg.d}_dataseed{args.data_seed}_seed{cfg.seed}.csv")

    def cb(k, theta, f, g, alpha, phase="main", **extras):
        logger.log(k, f, g, alpha, phase=phase, **extras)


    if args.method == "gd":
        theta_star = gradient_descent(
            theta0, obj_fn,
            max_iters=cfg.max_iters,
            tol_grad=cfg.tol_grad,
            alpha0=cfg.alpha0,
            use_line_search=cfg.use_line_search,
            callback=lambda k,t,f,g,alpha,**extras: cb(k,t,f,g,alpha,"main",**extras)

        )

    elif args.method == "pgd_diag_gn":
        precond = DiagGNPreconditioner(model=model, template=template, X=X, y=y, damping=cfg.damping)
        theta_star = preconditioned_gd(
            theta0, obj_fn, precond,
            max_iters=cfg.max_iters,
            tol_grad=cfg.tol_grad,
            alpha0=cfg.alpha0,
            use_line_search=cfg.use_line_search,
            precond_refresh=cfg.precond_refresh,
            callback=lambda k,t,f,g,alpha,**extras: cb(k,t,f,g,alpha,"main",**extras)
        )

    elif args.method == "lbfgs":
        theta_star, _state = lbfgs_minimize(
            theta0, obj_fn,
            max_iters=cfg.max_iters,
            tol_grad=cfg.tol_grad,
            history_size=cfg.lbfgs_history_size,
            alpha0=cfg.alpha0,
            use_line_search=cfg.use_line_search,
            callback=lambda k,t,f,g,alpha,**extras: cb(k,t,f,g,alpha,"main",**extras),
            line_search="strong_wolfe"
        )
        
    elif args.method == "lbfgs_d":
        theta_star, _state = lbfgs_d_minimize(
            theta0, obj_fn,
            max_iters=cfg.max_iters,
            tol_grad=cfg.tol_grad,
            history_size=cfg.lbfgs_history_size,
            alpha0=cfg.alpha0,
            use_line_search=cfg.use_line_search,
            diag_damping=1e-8,
            callback=lambda k,t,f,g,alpha,**extras: cb(k,t,f,g,alpha,"main",**extras),
            line_search="strong_wolfe"
        )

    else:
        raise ValueError("Unknown method")

    logger.close()
    print("Done. Final loss:", float(obj_fn(theta_star)[0]))


if __name__ == "__main__":
    main()
