import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "gd": "GD",
    "pgd_diag_gn": "PGD (Diag GN precond)",
    "lbfgs": "L-BFGS",
    "lbfgs_d": "D-LBFGS",
    "gn_cg": "GN-CG",
}

# For log-plots (must be > 0)
LOSS_FLOOR = 1e-12
GRAD_FLOOR = 1e-12


# ---------------------------
# Loading / cleaning
# ---------------------------

def load_runs(dataset: str, m: int, d: int, seeds: List[int], methods: List[str]) -> pd.DataFrame:
    """
    New naming scheme only:
      {dataset}_{method}_m{m}_d{d}_dataseed{X}_seed{seed}.csv

    We search in results/ recursively and take the first match if multiple dataseeds exist.
    """
    root = Path("results")
    if not root.exists():
        raise RuntimeError("results/ folder not found. Run from the project root.")

    frames = []
    missing = []

    for seed in seeds:
        for method in methods:
            pattern = f"{dataset}_{method}_m{m}_d{d}_dataseed*_seed{seed}.csv"
            hits = sorted(root.glob(f"**/{pattern}"), key=lambda p: str(p))

            if not hits:
                missing.append((dataset, method, seed, pattern))
                continue

            path = hits[0]  # if you have multiple dataseeds, this picks the first deterministically
            df = pd.read_csv(path)
            df["seed"] = seed
            df["method"] = method
            df["run_id"] = f"{method}_seed{seed}"
            frames.append(df)

    if not frames:
        msg = ["No CSVs found for the requested selection (new naming scheme)."]
        if missing:
            msg.append("Examples of missing (dataset, method, seed, glob):")
            msg.extend([f"  {t}" for t in missing[:12]])
            if len(missing) > 12:
                msg.append(f"  ... and {len(missing) - 12} more")
        raise RuntimeError("\n".join(msg))

    return pd.concat(frames, ignore_index=True)



def clean_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Coerce numeric cols
    - Replace inf -> NaN
    - Ensure iter exists
    - Keep ONE row per (method, seed, iter), preferring alpha-present row (post-step)
    - Do NOT replace NaNs with 0 (breaks log plots). We'll clamp only at plot-time.
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = [
        "iter", "k",
        "loss", "grad_norm", "alpha",
        "elapsed_sec", "ls_evals",
        "p_norm", "step_norm", "gTp",
        "h0_min", "h0_mean", "h0_max",
        "cg_iters", "cg_rnorm",
        "cum_evals",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure we have "iter"
    if "iter" not in df.columns:
        if "k" in df.columns:
            df = df.rename(columns={"k": "iter"})
        else:
            raise RuntimeError("CSV must contain an iteration column (iter or k).")

    # Stable sorting
    sort_cols = [c for c in ["method", "seed", "iter", "elapsed_sec"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Keep one row per iter (prefer alpha-present row)
    if "alpha" in df.columns:
        df["_has_alpha"] = df["alpha"].notna().astype(int)
        df = df.sort_values(
            ["method", "seed", "iter", "_has_alpha"] + (["elapsed_sec"] if "elapsed_sec" in df.columns else [])
        )
        df = df.groupby(["method", "seed", "iter"], as_index=False).tail(1)
        df = df.drop(columns=["_has_alpha"])
    else:
        df = df.groupby(["method", "seed", "iter"], as_index=False).tail(1)

    return df


def add_work_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define a simple "work proxy" based on objective/grad evals.
    By your logger, ls_evals counts extra obj_fn evals inside line search.
    Each iteration does 1 main eval + ls_evals line-search evals.
    """
    df = df.copy()
    if "ls_evals" not in df.columns:
        df["ls_evals"] = 0.0
    df["evals_this_iter"] = df["ls_evals"].fillna(0).astype(float) + 1.0
    df["cum_evals"] = df.groupby(["method", "seed"])["evals_this_iter"].cumsum()
    return df


def forward_fill_to_max_iter(df: pd.DataFrame, cols_to_ffill: List[str]) -> pd.DataFrame:
    """
    For median/IQR vs iteration, it's often nicer (and fairer across seeds) to
    extend each run to the global max iteration by forward-filling the last value.
    That avoids "shrinking sample size" artifacts after early stopping.

    Only forward-fill the columns you explicitly request (loss, grad_norm).
    """
    df = df.copy()
    max_iter = int(df["iter"].max())

    out_frames = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g = g.sort_values("iter").set_index("iter")
        idx = pd.RangeIndex(0, max_iter + 1)
        g2 = g.reindex(idx)

        # Keep identifiers
        g2["method"] = method
        g2["seed"] = seed
        g2["run_id"] = f"{method}_seed{seed}"

        # Forward fill selected columns
        for c in cols_to_ffill:
            if c in g2.columns:
                g2[c] = g2[c].ffill()

        # elapsed_sec and cum_evals should NOT be forward-filled (not meaningful)
        g2 = g2.reset_index().rename(columns={"index": "iter"})
        out_frames.append(g2)

    return pd.concat(out_frames, ignore_index=True)


# ---------------------------
# Robust aggregation + plotting helpers
# ---------------------------

def clamp_pos(x: np.ndarray, floor: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.where(x > floor, x, floor)
    return x


def agg_quantiles_by_x(df: pd.DataFrame, x_col: str, y_col: str,
                       qlo: float = 0.25, qhi: float = 0.75) -> pd.DataFrame:
    """
    Aggregate across seeds at each x for each method: median + IQR.
    """
    g = df[[ "method", x_col, y_col ]].dropna(subset=[x_col, y_col]).copy()
    g[x_col] = g[x_col].astype(float)

    out = (
        g.groupby(["method", x_col])[y_col]
         .agg(
            median="median",
            qlo=lambda s: s.quantile(qlo),
            qhi=lambda s: s.quantile(qhi),
            count="count",
         )
         .reset_index()
    )
    return out


def plot_median_iqr(ax, agg: pd.DataFrame, x_col: str, y_label: str,
                    title: str, logy: bool = False, legend: bool = True):
    for method in sorted(agg["method"].unique()):
        sub = agg[agg["method"] == method].sort_values(x_col)
        ax.plot(sub[x_col], sub["median"], label=METHOD_LABELS.get(method, method))
        ax.fill_between(sub[x_col], sub["qlo"], sub["qhi"], alpha=0.2)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if legend:
        ax.legend()


def plot_per_seed_lines(ax, df: pd.DataFrame, x_col: str, y_col: str,
                        title: str, y_label: str, logy: bool = False):
    """
    Thin lines per seed, one method at a time, with legend only once per method.
    Useful for visualizing variability and outliers.
    """
    for method in sorted(df["method"].unique()):
        df_m = df[df["method"] == method]
        first = True
        for seed, g in df_m.groupby("seed"):
            g = g.sort_values(x_col)
            ax.plot(
                g[x_col], g[y_col],
                alpha=0.25 if df_m["seed"].nunique() > 1 else 1.0,
                label=METHOD_LABELS.get(method, method) if first else None,
            )
            first = False
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()


def step_interpolate(x_src: np.ndarray, y_src: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """
    Piecewise-constant interpolation (hold-last-value).
    Assumes x_src is increasing.
    For x_grid before first x_src, uses first y.
    For x_grid after last x_src, uses last y.
    """
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    # Drop NaNs
    m = np.isfinite(x_src) & np.isfinite(y_src)
    x_src = x_src[m]
    y_src = y_src[m]
    if len(x_src) == 0:
        return np.full_like(x_grid, np.nan, dtype=float)

    order = np.argsort(x_src)
    x_src = x_src[order]
    y_src = y_src[order]

    # indices where each grid point would be inserted
    idx = np.searchsorted(x_src, x_grid, side="right") - 1
    idx = np.clip(idx, 0, len(x_src) - 1)
    return y_src[idx]


def median_iqr_on_common_grid(df: pd.DataFrame, x_col: str, y_col: str,
                              n_grid: int = 200,
                              qlo: float = 0.25, qhi: float = 0.75) -> pd.DataFrame:
    """
    For x_col that differs across runs (elapsed_sec, cum_evals), we:
      - build a common grid per method up to the minimum of max-x across seeds
      - step-interpolate each run onto that grid
      - compute median/IQR across seeds on that grid
    """
    out_frames = []
    for method, g_method in df.groupby("method"):
        runs = []
        max_xs = []

        for (seed), g in g_method.groupby("seed"):
            g = g.sort_values(x_col)
            xs = g[x_col].to_numpy(dtype=float)
            ys = g[y_col].to_numpy(dtype=float)
            # Keep only finite xs
            m = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[m]
            ys = ys[m]
            if len(xs) < 2:
                continue
            max_xs.append(xs.max())
            runs.append((xs, ys))

        if len(runs) == 0:
            continue

        x_max_common = float(np.min(max_xs))  # ensures all seeds contribute over the grid
        if x_max_common <= 0:
            continue

        x_grid = np.linspace(0.0, x_max_common, n_grid)

        Y = []
        for xs, ys in runs:
            y_grid = step_interpolate(xs, ys, x_grid)
            Y.append(y_grid)
        Y = np.vstack(Y)  # (n_seeds, n_grid)

        median = np.nanmedian(Y, axis=0)
        ql = np.nanquantile(Y, qlo, axis=0)
        qh = np.nanquantile(Y, qhi, axis=0)

        tmp = pd.DataFrame({
            "method": method,
            x_col: x_grid,
            "median": median,
            "qlo": ql,
            "qhi": qh,
            "count": np.sum(np.isfinite(Y), axis=0),
        })
        out_frames.append(tmp)

    if not out_frames:
        return pd.DataFrame(columns=["method", x_col, "median", "qlo", "qhi", "count"])
    return pd.concat(out_frames, ignore_index=True)


def compute_final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final loss, final grad_norm, final time, final work per (method, seed).
    """
    rows = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g = g.sort_values("iter")
        # final non-nan loss
        loss = g["loss"].dropna()
        grad = g["grad_norm"].dropna() if "grad_norm" in g.columns else pd.Series(dtype=float)

        rows.append({
            "method": method,
            "seed": seed,
            "final_iter": int(g["iter"].max()),
            "final_loss": float(loss.iloc[-1]) if len(loss) else np.nan,
            "final_grad_norm": float(grad.iloc[-1]) if len(grad) else np.nan,
            "final_time_sec": float(g["elapsed_sec"].dropna().iloc[-1]) if "elapsed_sec" in g.columns and g["elapsed_sec"].notna().any() else np.nan,
            "final_work": float(g["cum_evals"].dropna().iloc[-1]) if "cum_evals" in g.columns and g["cum_evals"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def time_to_target(df: pd.DataFrame, target_loss: float) -> pd.DataFrame:
    """
    For each (method, seed): first elapsed_sec where loss <= target.
    NaN if never reached.
    """
    rows = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g = g.sort_values("elapsed_sec") if "elapsed_sec" in g.columns else g.sort_values("iter")
        hit = g[g["loss"] <= target_loss]
        t = float(hit["elapsed_sec"].iloc[0]) if ("elapsed_sec" in g.columns and len(hit) > 0) else np.nan
        rows.append({"method": method, "seed": seed, "time_to_target": t})
    return pd.DataFrame(rows)


def work_to_target(df: pd.DataFrame, target_loss: float) -> pd.DataFrame:
    """
    For each (method, seed): first cum_evals where loss <= target.
    NaN if never reached.
    """
    rows = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g = g.sort_values("cum_evals") if "cum_evals" in g.columns else g.sort_values("iter")
        hit = g[g["loss"] <= target_loss]
        w = float(hit["cum_evals"].iloc[0]) if ("cum_evals" in g.columns and len(hit) > 0) else np.nan
        rows.append({"method": method, "seed": seed, "work_to_target": w})
    return pd.DataFrame(rows)


def boxplot_by_method(ax, df: pd.DataFrame, value_col: str, title: str, y_label: str, logy: bool = False):
    methods = sorted(df["method"].unique())
    data = [df[df["method"] == m][value_col].dropna().to_numpy(dtype=float) for m in methods]
    ax.boxplot(data, labels=[METHOD_LABELS.get(m, m) for m in methods], showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=15)
    if logy:
        ax.set_yscale("log")


# ---------------------------
# Tables for paper
# ---------------------------

def _label_method(m: str) -> str:
    return METHOD_LABELS.get(m, m)

def _iqr_str(q25: float, q75: float) -> str:
    if not np.isfinite(q25) or not np.isfinite(q75):
        return ""
    return f"[{q25:.4g}, {q75:.4g}]"

def _agg_stats(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    }

def save_table_png(df: pd.DataFrame, out_png: str, title: str = "", font_size: int = 10):
    """
    Render a DataFrame as a PNG image using matplotlib.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    df_disp = df.copy().fillna("")
    df_disp = df_disp.astype(str)

    nrows, ncols = df_disp.shape
    fig_w = max(8, 1.0 * ncols + 2)
    fig_h = max(2.0, 0.35 * nrows + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns.tolist(),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.2)

    if title:
        ax.set_title(title, pad=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_table(df: pd.DataFrame, out_csv: str, out_png: str, title: str = ""):
    """
    Save table as CSV + PNG (no LaTeX).
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    save_table_png(df, out_png, title=title)


def iter_to_target(df: pd.DataFrame, target_loss: float) -> pd.DataFrame:
    """
    First iteration where loss <= target for each (method, seed).
    """
    rows = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g = g.sort_values("iter")
        hit = g[g["loss"] <= target_loss]
        it = int(hit["iter"].iloc[0]) if len(hit) else np.nan
        rows.append({"method": method, "seed": seed, "iter_to_target": it})
    return pd.DataFrame(rows)

def best_loss_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best (minimum) loss achieved, and iteration/time/work where it occurs, per (method, seed).
    """
    rows = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g = g.sort_values("iter")
        loss = pd.to_numeric(g["loss"], errors="coerce")
        if loss.notna().any():
            idx = loss.idxmin()
            rows.append({
                "method": method,
                "seed": seed,
                "best_loss": float(loss.loc[idx]),
                "iter_at_best": float(g.loc[idx, "iter"]) if "iter" in g.columns else np.nan,
                "time_at_best": float(g.loc[idx, "elapsed_sec"]) if "elapsed_sec" in g.columns else np.nan,
                "work_at_best": float(g.loc[idx, "cum_evals"]) if "cum_evals" in g.columns else np.nan,
            })
        else:
            rows.append({"method": method, "seed": seed, "best_loss": np.nan,
                         "iter_at_best": np.nan, "time_at_best": np.nan, "work_at_best": np.nan})
    return pd.DataFrame(rows)

def loss_at_iter_budgets(df_iter_filled: pd.DataFrame, budgets: List[int]) -> pd.DataFrame:
    """
    Median/IQR loss at fixed iteration budgets (complements loss-vs-iter plots).
    df_iter_filled should be the forward-filled dataframe (like your df_iter).
    """
    rows = []
    for b in budgets:
        sub = df_iter_filled[df_iter_filled["iter"] == b]
        if sub.empty:
            continue
        for method, g in sub.groupby("method"):
            st = _agg_stats(g["loss"])
            rows.append({
                "method": _label_method(method),
                "budget_iter": b,
                "loss_median": st["median"],
                "loss_iqr": _iqr_str(st["q25"], st["q75"]),
                "loss_mean": st["mean"],
                "loss_std": st["std"],
                "n": int(g["seed"].nunique()),
            })
    return pd.DataFrame(rows).sort_values(["budget_iter", "method"])

def auc_log_loss_over_iter(df_iter_filled: pd.DataFrame) -> pd.DataFrame:
    """
    AUC of log10(loss) vs iteration for each run; then you can summarize across seeds.
    Lower is better (faster convergence).
    """
    rows = []
    for (method, seed), g in df_iter_filled.groupby(["method", "seed"]):
        g = g.sort_values("iter")
        x = g["iter"].to_numpy(dtype=float)
        y = np.log10(clamp_pos(g["loss"].to_numpy(dtype=float), LOSS_FLOOR))
        if len(x) >= 2 and np.isfinite(y).any():
            auc = float(np.trapz(y, x))
        else:
            auc = np.nan
        rows.append({"method": method, "seed": seed, "auc_log10_loss_iter": auc})
    return pd.DataFrame(rows)

def auc_log_loss_over_common_grid(df: pd.DataFrame, x_col: str, y_col: str, n_grid: int = 300) -> pd.DataFrame:
    """
    AUC of log10(loss) vs x_col (elapsed_sec or cum_evals), per run,
    computed on a method-specific common grid so seeds are comparable.
    """
    rows = []
    for method, g_method in df.groupby("method"):
        runs = []
        max_xs = []
        for seed, g in g_method.groupby("seed"):
            g = g.sort_values(x_col)
            xs = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
            ys = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[m], ys[m]
            if len(xs) < 2:
                continue
            max_xs.append(xs.max())
            runs.append((seed, xs, ys))

        if not runs:
            continue

        x_max_common = float(np.min(max_xs))
        if x_max_common <= 0:
            continue
        x_grid = np.linspace(0.0, x_max_common, n_grid)

        for seed, xs, ys in runs:
            y_grid = step_interpolate(xs, ys, x_grid)
            # integrate y over x
            auc = float(np.trapz(y_grid, x_grid)) if np.isfinite(y_grid).any() else np.nan
            rows.append({"method": method, "seed": seed, f"auc_{y_col}_vs_{x_col}": auc, "x_max_common": x_max_common})

    return pd.DataFrame(rows)

def summarize_by_method(run_level_df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """
    Summarize run-level metrics across seeds into median/IQR/mean/std per method.
    """
    rows = []
    for method, g in run_level_df.groupby("method"):
        row = {"method": _label_method(method), "n": int(g["seed"].nunique())}
        for c in value_cols:
            st = _agg_stats(g[c]) if c in g.columns else {"mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
            row[f"{c}_median"] = st["median"]
            row[f"{c}_iqr"] = _iqr_str(st["q25"], st["q75"])
            row[f"{c}_mean"] = st["mean"]
            row[f"{c}_std"] = st["std"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method")

def line_search_effort_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-run LS effort: total ls_evals, mean per iter, then summarized per method.
    """
    per_run = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        ls = pd.to_numeric(g.get("ls_evals", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        iters = pd.to_numeric(g["iter"], errors="coerce").dropna()
        n_steps = int(iters.max() + 1) if len(iters) else np.nan
        per_run.append({
            "method": method,
            "seed": seed,
            "ls_total": float(ls.sum()) if len(ls) else 0.0,
            "ls_mean_per_iter": float(ls.mean()) if len(ls) else 0.0,
            "iters": n_steps,
        })
    per_run = pd.DataFrame(per_run)
    return summarize_by_method(per_run, ["ls_total", "ls_mean_per_iter"])

def targets_table(df: pd.DataFrame, targets: List[float]) -> pd.DataFrame:
    """
    For each target: success rate + median time/work/iter-to-target among successful seeds.
    """
    out = []
    for targ in targets:
        itdf = iter_to_target(df, targ)
        tdf = time_to_target(df, targ) if "elapsed_sec" in df.columns else None
        wdf = work_to_target(df, targ) if "cum_evals" in df.columns else None

        for method in sorted(df["method"].unique()):
            sub_i = itdf[itdf["method"] == method]
            total = len(sub_i)
            succ_mask = sub_i["iter_to_target"].notna()
            succ = int(succ_mask.sum())
            rate = succ / max(1, total)

            row = {
                "target_loss": targ,
                "method": _label_method(method),
                "success_rate": rate,
                "n_success": succ,
                "n_total": total,
            }

            # Iter-to-target stats (successful only)
            st_i = _agg_stats(sub_i.loc[succ_mask, "iter_to_target"])
            row["iter_to_target_median"] = st_i["median"]
            row["iter_to_target_iqr"] = _iqr_str(st_i["q25"], st_i["q75"])

            # Time/work stats (successful only)
            if tdf is not None:
                sub_t = tdf[tdf["method"] == method]
                st_t = _agg_stats(sub_t["time_to_target"])
                row["time_to_target_median"] = st_t["median"]
                row["time_to_target_iqr"] = _iqr_str(st_t["q25"], st_t["q75"])
            if wdf is not None:
                sub_w = wdf[wdf["method"] == method]
                st_w = _agg_stats(sub_w["work_to_target"])
                row["work_to_target_median"] = st_w["median"]
                row["work_to_target_iqr"] = _iqr_str(st_w["q25"], st_w["q75"])

            out.append(row)

    return pd.DataFrame(out).sort_values(["target_loss", "method"])

def write_paper_tables(df_raw: pd.DataFrame, df_iter_filled: pd.DataFrame, outdir: str, targets: List[float]):
    """
    Writes:
      - final_summary (accuracy + time + work)
      - best_loss_summary (best achieved, and where)
      - auc_summary (convergence speed score)
      - loss_at_budgets (anytime performance)
      - targets (success + time/work/iter to hit loss)
      - line_search_effort (overhead)
    """
    tables_dir = os.path.join(outdir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Run-level summaries
    final_df = compute_final_metrics(df_raw)
    final_summary = summarize_by_method(final_df, ["final_loss", "final_grad_norm", "final_time_sec", "final_work"])
    _write_table(final_summary,
                 os.path.join(tables_dir, "final_summary.csv"),
                 os.path.join(tables_dir, "final_summary.png"),
                 title="Final metrics summary")

    best_df = best_loss_metrics(df_raw)
    best_summary = summarize_by_method(best_df, ["best_loss", "iter_at_best", "time_at_best", "work_at_best"])
    _write_table(best_summary,
                 os.path.join(tables_dir, "best_loss_summary.csv"),
                 os.path.join(tables_dir, "best_loss_summary.png"),
                 title="Best loss summary")

    # AUC scores (lower = better)
    auc_iter = auc_log_loss_over_iter(df_iter_filled)
    auc_iter_sum = summarize_by_method(auc_iter, ["auc_log10_loss_iter"])

    auc_tables = [auc_iter_sum]
    if "elapsed_sec" in df_raw.columns:
        df_t = df_raw.copy()
        df_t["loss_log"] = np.log10(clamp_pos(df_t["loss"].to_numpy(dtype=float), LOSS_FLOOR))
        auc_time = auc_log_loss_over_common_grid(df_t, "elapsed_sec", "loss_log", n_grid=300)
        if not auc_time.empty:
            auc_time_sum = summarize_by_method(auc_time, ["auc_loss_log_vs_elapsed_sec"])
            auc_tables.append(auc_time_sum)

    if "cum_evals" in df_raw.columns:
        df_w = df_raw.copy()
        df_w["loss_log"] = np.log10(clamp_pos(df_w["loss"].to_numpy(dtype=float), LOSS_FLOOR))
        auc_work = auc_log_loss_over_common_grid(df_w, "cum_evals", "loss_log", n_grid=300)
        if not auc_work.empty:
            auc_work_sum = summarize_by_method(auc_work, ["auc_loss_log_vs_cum_evals"])
            auc_tables.append(auc_work_sum)

    # Merge AUC summaries by method label
    auc_merged = auc_tables[0]
    for t in auc_tables[1:]:
        auc_merged = auc_merged.merge(t, on=["method", "n"], how="outer")
    _write_table(auc_merged,
                 os.path.join(tables_dir, "auc_summary.csv"),
                 os.path.join(tables_dir, "auc_summary.png"),
                 title="AUC summary (lower is better)")

    # Anytime performance: loss at fixed iter budgets
    budgets = [50, 100, 200, 400]
    budget_tbl = loss_at_iter_budgets(df_iter_filled, budgets)
    _write_table(budget_tbl,
                 os.path.join(tables_dir, "loss_at_iter_budgets.csv"),
                 os.path.join(tables_dir, "loss_at_iter_budgets.png"),
                 title="Loss at iteration budgets")

    # Targets table (success + time/work/iter)
    targ_tbl = targets_table(df_raw, targets)
    _write_table(targ_tbl,
                 os.path.join(tables_dir, "targets_summary.csv"),
                 os.path.join(tables_dir, "targets_summary.png"),
                 title="Target hit rates and cost")

    # Line search overhead
    ls_tbl = line_search_effort_table(df_raw)
    _write_table(ls_tbl,
                 os.path.join(tables_dir, "line_search_effort.csv"),
                 os.path.join(tables_dir, "line_search_effort.png"),
                 title="Line search effort")

    print("Wrote tables to:", tables_dir)


# ---------------------------
# Main plots
# ---------------------------

def make_plots(df: pd.DataFrame, outdir: str, title_prefix: str,
               targets: List[float]):
    os.makedirs(outdir, exist_ok=True)

    df = clean_runs(df)
    df = add_work_columns(df)

    # For iteration-based aggregation, forward-fill loss/grad to max iter
    df_iter = forward_fill_to_max_iter(df, cols_to_ffill=["loss", "grad_norm"])

    # Clamp only for log-based plotting/statistics
    df_iter = df_iter.copy()
    df_iter["loss_log"] = clamp_pos(df_iter["loss"].to_numpy(), LOSS_FLOOR)
    if "grad_norm" in df_iter.columns:
        df_iter["grad_log"] = clamp_pos(df_iter["grad_norm"].to_numpy(), GRAD_FLOOR)

    # 1) Loss vs Iteration (Median + IQR), linear
    fig, ax = plt.subplots()
    agg = agg_quantiles_by_x(df_iter, "iter", "loss")
    plot_median_iqr(
        ax, agg, "iter",
        y_label="Loss",
        title=f"{title_prefix}: Loss vs Iteration (median ± IQR)",
        logy=False,
    )
    # Optional: make y-limits more interpretable (robust)
    y95 = np.nanpercentile(df_iter["loss"].to_numpy(dtype=float), 95)
    ax.set_ylim(0.0, max(1.0, min(300.0, 1.1 * y95)))
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_vs_iter_median.png"), dpi=200)
    plt.close(fig)

    # 2) Loss vs Iteration (Median + IQR), log
    fig, ax = plt.subplots()
    agg = agg_quantiles_by_x(df_iter, "iter", "loss_log")
    plot_median_iqr(
        ax, agg, "iter",
        y_label="Loss (log scale)",
        title=f"{title_prefix}: Loss vs Iteration (median ± IQR)",
        logy=True,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_vs_iter_median_log.png"), dpi=200)
    plt.close(fig)

    # 3) Loss vs Iteration per-seed (log) to show outliers
    fig, ax = plt.subplots()
    df_seed = df_iter.copy()
    df_seed["loss_log"] = clamp_pos(df_seed["loss"].to_numpy(), LOSS_FLOOR)
    plot_per_seed_lines(
        ax, df_seed, "iter", "loss_log",
        title=f"{title_prefix}: Loss vs Iteration (per-seed, log)",
        y_label="Loss (log scale)",
        logy=True,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_vs_iter_per_seed_log.png"), dpi=200)
    plt.close(fig)

    # 4) Grad norm vs Iteration (Median + IQR), log
    if "grad_norm" in df_iter.columns:
        fig, ax = plt.subplots()
        agg = agg_quantiles_by_x(df_iter, "iter", "grad_log")
        plot_median_iqr(
            ax, agg, "iter",
            y_label="||grad|| (log scale)",
            title=f"{title_prefix}: Grad Norm vs Iteration (median ± IQR)",
            logy=True,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "gradnorm_vs_iter_median_log.png"), dpi=200)
        plt.close(fig)

    # 5) Loss vs Time (Median + IQR), log, with interpolation grid
    if "elapsed_sec" in df.columns:
        df_time = df.copy()
        df_time["loss_log"] = clamp_pos(df_time["loss"].to_numpy(), LOSS_FLOOR)

        fig, ax = plt.subplots()
        agg = median_iqr_on_common_grid(df_time, "elapsed_sec", "loss_log", n_grid=250)
        plot_median_iqr(
            ax, agg, "elapsed_sec",
            y_label="Loss (log scale)",
            title=f"{title_prefix}: Loss vs Time (median ± IQR)",
            logy=True,
        )
        ax.set_xlabel("Elapsed time (sec)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "loss_vs_time_median_log.png"), dpi=200)
        plt.close(fig)

        # Per-seed time plot (log) to visualize variability/outliers
        fig, ax = plt.subplots()
        plot_per_seed_lines(
            ax, df_time, "elapsed_sec", "loss_log",
            title=f"{title_prefix}: Loss vs Time (per-seed, log)",
            y_label="Loss (log scale)",
            logy=True,
        )
        ax.set_xlabel("Elapsed time (sec)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "loss_vs_time_per_seed_log.png"), dpi=200)
        plt.close(fig)

    # 6) Loss vs Work proxy (Median + IQR), log
    if "cum_evals" in df.columns:
        df_work = df.copy()
        df_work["loss_log"] = clamp_pos(df_work["loss"].to_numpy(), LOSS_FLOOR)

        fig, ax = plt.subplots()
        agg = median_iqr_on_common_grid(df_work, "cum_evals", "loss_log", n_grid=250)
        plot_median_iqr(
            ax, agg, "cum_evals",
            y_label="Loss (log scale)",
            title=f"{title_prefix}: Loss vs Work (median ± IQR)",
            logy=True,
        )
        ax.set_xlabel("Cumulative obj/grad evals (proxy)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "loss_vs_work_median_log.png"), dpi=200)
        plt.close(fig)

    # 7) Final loss distribution (boxplot) across seeds (captures robustness + outliers)
    final_df = compute_final_metrics(df)
    fig, ax = plt.subplots()
    boxplot_by_method(
        ax, final_df, "final_loss",
        title=f"{title_prefix}: Final Loss Distribution Across Seeds",
        y_label="Final loss",
        logy=False,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "final_loss_box.png"), dpi=200)
    plt.close(fig)

    # 8) Time-to-target and Work-to-target boxplots + success rates
    for targ in targets:
        if "elapsed_sec" in df.columns:
            tdf = time_to_target(df, targ)
            fig, ax = plt.subplots()
            boxplot_by_method(
                ax, tdf, "time_to_target",
                title=f"{title_prefix}: Time to Reach Loss ≤ {targ} (Across Seeds)",
                y_label="Time (sec)",
                logy=False,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"time_to_loss_{targ:g}.png"), dpi=200)
            plt.close(fig)

        if "cum_evals" in df.columns:
            wdf = work_to_target(df, targ)
            fig, ax = plt.subplots()
            boxplot_by_method(
                ax, wdf, "work_to_target",
                title=f"{title_prefix}: Work to Reach Loss ≤ {targ} (Across Seeds)",
                y_label="Cumulative evals (proxy)",
                logy=False,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"work_to_loss_{targ:g}.png"), dpi=200)
            plt.close(fig)

        # Success rate (fraction of seeds that reach target)
        # Save as a simple bar chart for each target
        success_rows = []
        for method in sorted(df["method"].unique()):
            hit = 0
            total = 0
            for seed, g in df[df["method"] == method].groupby("seed"):
                total += 1
                if (g["loss"] <= targ).any():
                    hit += 1
            success_rows.append((method, hit / max(1, total)))

        fig, ax = plt.subplots()
        methods = [m for m, _ in success_rows]
        rates = [r for _, r in success_rows]
        ax.bar(range(len(methods)), rates)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Success rate")
        ax.set_title(f"{title_prefix}: Success Rate for Loss ≤ {targ}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"success_rate_loss_{targ:g}.png"), dpi=200)
        plt.close(fig)

    # 9) Line search effort vs iteration (median + IQR)
    if "ls_evals" in df_iter.columns:
        fig, ax = plt.subplots()
        agg = agg_quantiles_by_x(df_iter, "iter", "ls_evals")
        plot_median_iqr(
            ax, agg, "iter",
            y_label="Line search evals (median ± IQR)",
            title=f"{title_prefix}: Line Search Effort vs Iteration",
            logy=False,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ls_evals_vs_iter_median.png"), dpi=200)
        plt.close(fig)

    # 10) D-LBFGS diagonal stats (if present): h0_max over iter (median + IQR)
    if "h0_max" in df_iter.columns:
        fig, ax = plt.subplots()
        # clamp for log if desired; here keep linear, but you can swap to log if it explodes
        agg = agg_quantiles_by_x(df_iter, "iter", "h0_max")
        plot_median_iqr(
            ax, agg, "iter",
            y_label="H0 diag max (median ± IQR)",
            title=f"{title_prefix}: D-LBFGS H0 Diagonal Max vs Iteration",
            logy=False,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "dlbfgs_h0_max_vs_iter.png"), dpi=200)
        plt.close(fig)

    # 11) GN-CG diagnostics if present
    if "cg_iters" in df_iter.columns and (df_iter["cg_iters"].notna().any()):
        fig, ax = plt.subplots()
        agg = agg_quantiles_by_x(df_iter, "iter", "cg_iters")
        plot_median_iqr(
            ax, agg, "iter",
            y_label="CG iterations (median ± IQR)",
            title=f"{title_prefix}: GN-CG Inner Iterations vs Outer Iteration",
            logy=False,
        )
        ax.set_xlabel("GN outer iteration")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "gn_cg_iters_median.png"), dpi=200)
        plt.close(fig)
    
    # --- Paper tables ---
    write_paper_tables(df, df_iter, outdir, targets)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sine", "ill_cond"])
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--methods", type=str, default="gd,pgd_diag_gn,lbfgs,lbfgs_d,gn_cg")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--targets", type=str, default="100,10,5,2",
                    help="Comma-separated loss targets for time/work-to-target plots.")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    targets = [float(t.strip()) for t in args.targets.split(",") if t.strip()]

    df = load_runs(args.dataset, args.m, args.d, seeds, methods)
    title_prefix = {"sine": "Sine", "ill_cond": "Ill-Conditioned"}.get(args.dataset, args.dataset)
    outdir = os.path.join(args.outdir, f"{args.dataset}_m{args.m}_d{args.d}")

    make_plots(df, outdir, title_prefix, targets)
    print("Wrote plots to:", outdir)


if __name__ == "__main__":
    main()
