### Setup
pip install -r requirements.txt

### ML Optimization Project

This repository contains training and plotting utilities used to run and analyze optimization experiments on small neural nets.

**Setup**

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Training (run.py)**

Train a single run and write a CSV log to `results/`:

```bash
python run.py --dataset sine --method gd --m 1000 --d 1 --iters 300 --seed 0 --data_seed 0
```

Available `--dataset` values: `sine`, `ill_cond`.

Available `--method` values: `gd`, `pgd_diag_gn`, `lbfgs`, `pgd_lbfgs_precond`, `gn_cg`, `lbfgs_d`.

The training script writes CSV logs named like:

```
results/{dataset}_{method}_m{m}_d{d}_dataseed{data_seed}_seed{seed}.csv
```

Run multiple seeds (example):

```bash
for s in 0 1 2 3 4; do
	python run.py --dataset sine --method gd --m 1000 --d 1 --iters 300 --seed $s --data_seed 0
done
```

**Plotting & Tables (plot_results.py)**

`plot_results.py` reads CSV logs from `results/`, aggregates across seeds/methods and writes plots and tables.

Key arguments:
- `--dataset` (required): `sine` or `ill_cond`
- `--m`, `--d`: dataset dimensions
- `--seeds`: comma-separated seed list (e.g. `0,1,2,3,4`)
- `--methods`: comma-separated methods to include (e.g. `gd,pgd_diag_gn,lbfgs`)
- `--outdir`: output directory (default `figures`)
- `--targets`: comma-separated loss targets for time/work-to-target analysis

Example (generate plots + tables for many seeds and methods):

```bash
python plot_results.py \
	--dataset sine --m 1000 --d 10 \
	--seeds 0,1,2,3,4,5,6,7,8,9 \
	--methods gd,pgd_diag_gn,lbfgs,lbfgs_d \
	--outdir figures
```

Another example (ill-conditioned dataset):

```bash
python plot_results.py --dataset ill_cond --m 1000 --d 10 --seeds 0,1,2,3,4,5,6,7,8,9 --methods gd,pgd_diag_gn,lbfgs,lbfgs_d --outdir figures
```

Output structure:

- Plots are saved under `{outdir}/{dataset}_m{m}_d{d}/` (PNG files).
- Paper tables are written to `{outdir}/tables/` as both CSV and PNG (table images).

If `results/` is empty or missing CSVs for the requested selection, `plot_results.py` will raise an error. Run `run.py` first to produce CSV logs.

**Notes**

- A `.gitignore` has been added to exclude caches, virtualenvs, model files (`*.pt`, `*.pkl`), logs and large output folders like `figures/` and `results/`.
- Run all commands from the project root so the scripts can find `results/` and write outputs to the expected locations.
