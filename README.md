# RiemannianNystrom

Minimal reproduction code for the numerical experiments in the paper section
`Numerical experiments`:

- HDM05 principal geodesic analysis (PGA) table.
- RRNCN on the SPD manifold.
- Transported sketching on the Grassmann manifold.

This repository intentionally excludes raw data, generated results, caches, and
experiments that are not used in the paper.

## Environment

Use Python 3.9 or newer.

Clone this repository with Git (rather than downloading a ZIP), since the new
Grassmann runner records the repository URL, commit, and source hashes.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## HDM05 PGA table

The paper uses the preprocessed SPD covariance-matrix version of HDM05 from the
supplementary material of Brooks et al. Place the `.npy` matrices under:

```text
data/raw/hdm05_extracted/hdm05/
```

Each file is expected to be named like:

```text
id_length_label.npy
```

Run the table reproduction:

```bash
scripts/run_hdm05_table.sh
```

The underlying PGA driver is `PGA/run_hdm05_downstream.py`, with helper routines
in `PGA/pga_utils.py`.

If the dataset lives elsewhere:

```bash
HDM05_RAW_DIR=/path/to/hdm05 scripts/run_hdm05_table.sh
```

The script runs three classifier settings and writes:

- `results/hdm05_table_logreg_pairwise_storage/`
- `results/hdm05_table_svm_tuned/`
- `results/hdm05_table_mlp/`
- `results/hdm05_table_summary.csv`

The summary columns correspond to the paper table: logreg/SVM/MLP accuracy,
measured memory in MiB, and median pairwise Hotelling `T^2 x 1e3`.

## SPD RRNCN figure

Run:

```bash
scripts/run_spd_figure.sh
```

Outputs are written under:

```text
reports-spd/spd_nystrom_cubic_arc_newton_phase_probe_nogradsketch_d40_subset_l20_l80_l320/
```

The paper figure uses:

- `objective_vs_time_spd_nystrom_cubic_arc_newton_phase_probe_nogradsketch_d40_subset_l20_l80_l320_init0.eps`
- `nystrom_hessian_error_vs_iteration_spd_nystrom_cubic_arc_newton_phase_probe_nogradsketch_d40_subset_l20_l80_l320_init0_hesserr.eps`

## Grassmann transported-sketching figure

The revised experiment uses a materialized dense, non-diagonal symmetric matrix
of size `n=1024`, a Grassmann point with `p=20`, and sketch size `ell=40`.
It compares optimized fresh sketching (`T=1`) with transported sketching
(`T=2,4,8`). The transported variants reuse products of the fixed matrix with
the sketch directions while assembling the current reduced Hessian at every
iteration. No external dataset is needed.

Run the five paired seeds and three timing repetitions serially, then generate
the iteration/time figure:

```bash
scripts/run_grassmann_figure.sh
```

Outputs are written under:

```text
results/grassmann_dense_n1024/
reports-grassmann/dense_action_periods_n1024/
```

The paper figure uses:

- `dense_action_periods_n1024_p20_l40_T2_T4_T8_seed90_rep0_iteration.eps`
- `dense_action_periods_n1024_p20_l40_T2_T4_T8_seed90_rep0_time.eps`

The plots use the preselected seed 90, repetition 0, with legends `T=1,2,4,8`
and ordinate `f-f^star`; they are individual trajectories, not averaged curves.
Timing summaries use all complete paired runs, and runtimes depend on hardware
and numerical libraries. Existing outputs are never overwritten. See the
[dense transported-sketching instructions](experiments/transport_revision/dense_action_cache/README.md)
for tests, separate runner/report commands, and new output directories.

The earlier Grassmann implementation remains available under `code-grassmann`;
it is no longer the default experiment run by `scripts/run_grassmann_figure.sh`.

## Run everything

```bash
scripts/run_all.sh
```

The full HDM05 table run requires the external HDM05 dataset and can take a long
time. The repository does not redistribute that dataset.
