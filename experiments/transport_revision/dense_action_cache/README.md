# Dense transported-sketching experiment

This directory implements the revised Grassmann experiment. Run all commands
from the repository root after installing `requirements.txt` in a Git clone.
The existing `code-grassmann/rsncn` solver is retained; the experiment installs
dense-matrix kernels, full-polar transport, and a shared cubic subproblem solver.

## Configuration

The frozen configuration is `../plans/dense_action_periods_n1024.json`:

- Materialized dense, non-diagonal symmetric `A`, with `n=1024`, `p=20`, and
  sketch size `ell=40`. A random orthogonal change of basis generates `A`, but
  the solver receives only `A`, the initial point, and solver options, not the
  generating eigenbasis or eigenvalues.
- Optimized fresh sketching at `T=1` and the same action-cached transport
  implementation at `T=2,4,8`. Transport updates the products of the fixed `A`
  with the sketch directions. Every trial still assembles the current reduced
  Hessian; this is not a BFGS update or reuse of a previous Hessian.
- Five paired problem seeds (90--94), three timing repetitions per arm, and
  randomized serial execution. The target relative objective gap is `1e-6`.
  A separate symmetric eigensolve supplies the numerical objective reference.
- QR retraction, full-polar isometric transport, gradient insertion, and the
  same cubic solver and acceptance rules in all arms. `T` counts trial steps;
  rejected steps or poor model agreement can trigger earlier refreshes.

The plan retains the historical experiment protocol for traceability. Its
references to an earlier ablation describe the original study, not a dependency
on unpublished raw data. New runs generate their own records and source hashes.

## Run and plot

```bash
scripts/run_grassmann_figure.sh
```

This writes 60 raw records to `results/grassmann_dense_n1024` and the report,
CSV summaries, and EPS/PNG figures to
`reports-grassmann/dense_action_periods_n1024`. Do not run timing jobs
concurrently with tests, plotting, or other compute-intensive work.

To run each stage separately:

```bash
python experiments/transport_revision/dense_action_cache/run_action_cache_experiments.py \
  --plan experiments/transport_revision/plans/dense_action_periods_n1024.json \
  --out results/grassmann_dense_n1024

python experiments/transport_revision/tools/report_dense_action_periods.py \
  --results results/grassmann_dense_n1024 \
  --plan experiments/transport_revision/plans/dense_action_periods_n1024.json \
  --out reports-grassmann/dense_action_periods_n1024
```

Use new output directories for subsequent runs, for example:

```bash
GRASSMANN_RESULTS_DIR=results/grassmann_dense_repeat \
GRASSMANN_REPORT_DIR=reports-grassmann/dense_repeat \
scripts/run_grassmann_figure.sh
```

`EXPERIMENT_PYTHON` can select another Python executable. The reporter does not
require the earlier ablation records. If those records are available,
`--previous-results /path/to/earlier/ablation` enables an additional historical
input comparison; otherwise this check is explicitly reported as not requested.
Current-run source hashes, input pairing, failures, and target attainment are
checked in either case. Source files and the plan must remain unchanged between
running and reporting. Old local records with a different source inventory are
not silently relabeled as new runs.

The iteration/time figures use the preselected seed 90, repetition 0, with the
original red-family style, legends `T=1,2,4,8`, and ordinate `f-f^star`.
These are individual trajectories. Timing summaries first use all three valid
repetitions for each arm/seed and then compare paired seeds; incomplete or
failed runs remain visible. Loop timing excludes initialization, matrix
generation, and the reference eigensolve. Measured speedups depend on hardware
and numerical libraries and are not promised on other systems.

## Small correctness checks

Run these separately from performance experiments:

```bash
python -m unittest discover -s experiments/transport_revision -p 'test_cubic_verified.py' -v
python -m unittest discover -s experiments/transport_revision -p 'test_solver_reduced_core.py' -v
python -m unittest discover -s experiments/transport_revision -p 'test_geometry_isometric_fused.py' -v
python -m unittest discover -s experiments/transport_revision/dense -p 'test_kernels.py' -v
python -m unittest discover -s experiments/transport_revision/dense_opt -p 'test_kernels.py' -v
python -m unittest discover -s experiments/transport_revision/dense_action_cache -p 'test_action_cache.py' -v
python -m unittest discover -s experiments/transport_revision/tools -p 'test_report_dense_action_periods.py' -v
```

These test dense actions/current reduced Hessians, transport and gradient
updates, source/option consistency, the cubic subproblem, and reporting
portability. They do not assert a speedup or replace a full performance run.
The imported `reduced_core.py` retains the solver adapter's original reference
path; the dense runner overrides that path with the current dense Hessian
calculation, so the published dense experiment is not a diagonal problem.
