"""Frozen dense protocol with explicit direct/action-cache ablation labels.

case.action_cache_enabled=false: current direct A-products every trial.
case.action_cache_enabled=true: eligible fixed-A linear action updates.
Fresh T1 and transported T>1 otherwise share the SAME oracle, geometry,
core assembly, verified cubic solver, options and objective-gap evaluator.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))
from dense import run_dense_experiments as protocol
from dense_action_cache.kernels import ActionCacheOracle
from dense_action_cache.geometry import ActionCacheGeometry
from dense_opt.kernels import DenseBlockOracle
from geometry_isometric_fused import make_hooks as original_fused_hooks

CORE = "dense_fixed_A_action_cache_current"
_old_sources = protocol.source_map
_old_provenance = protocol.provenance
_old_instrument = protocol.instrument_dense
_old_run_one = protocol.run_one
_enable_updates = True
_optimized_direct_control = False


def source_map():
    result = _old_sources()
    for folder in (EXPERIMENT/"dense_opt", HERE):
        result.update({str(path.relative_to(protocol.REPO)): protocol.sha(path)
                       for path in sorted(folder.glob("*.py"))})
    return result


def provenance(plan_path):
    result = _old_provenance(plan_path)
    result.update(
        core_variant="fixed_dense_A_linear_action_cache_current_hessian",
        common_kernel_policy="Three primary ablation arms use ActionCacheOracle and the same current core assembly; action_cache_enabled selects fixed-A action updates. A separately labeled optimized_direct_control T1 arm uses the previous DenseBlockOracle+fused geometry to measure potential common-oracle overhead.",
        change_scope="New exact-in-real-arithmetic fixed-A linear-action-update variant, not merely contraction scheduling and not a claim for variable A(Y).",
        cache_contract="Fixed read-only dense A; exact private immutable frame provenance; full-polar M,C and gradient u,a,s shared with geometry; rank-one s_squared<0.25 triggers direct rebase without changing the frame.",
        cache_roundoff_policy="No finite-precision exactness claim. Diagnostic-only runs check direct A-actions and current core every trial; timing runs never include these checks.",
        phase_timing_note="Action propagation is included in transport/build phases; direct action rebases are included in current-core phase. End-to-end solver loop is the common performance measure.",
    )
    return result


@contextmanager
def instrument_dense(*, validate=False, chunk_size=16, workspace_bytes=128*1024**2):
    if _optimized_direct_control:
        oracle = DenseBlockOracle(chunk_size=chunk_size, workspace_bytes=workspace_bytes)
        hooks = original_fused_hooks(skip_transport_reorth=False)
    else:
        oracle = ActionCacheOracle(chunk_size=chunk_size, workspace_bytes=workspace_bytes,
                                   enable_updates=_enable_updates, diagnostic=validate)
        hooks = ActionCacheGeometry(oracle, skip_transport_reorth=False)
    previous_oracle, previous_hooks = protocol.DenseOracle, protocol.make_hooks
    protocol.DenseOracle = lambda **kwargs: oracle
    protocol.make_hooks = lambda **kwargs: hooks
    try:
        with _old_instrument(validate=validate, chunk_size=chunk_size, workspace_bytes=workspace_bytes) as state:
            phases, counts, checks, actual_oracle = state
            if _optimized_direct_control:
                checks["reduced_core_backend"] = "dense_large_block_current"
            else:
                checks["hessian_implementation"] = (
                "Exact-in-real-arithmetic CURRENT projected core using current S and current frame actions. "
                "Actions are actual dense A-products or fixed-A linear updates sharing geometry coefficients. "
                "No old Hessian, eigenbasis or generating spectrum. Cache-disabled arms rebase every core.")
                checks["hessian_count_semantics"] = (
                "current_core_columns counts every assembled current frame column. Actual dense products are "
                "recorded separately by action_rebase/action_transport/action_gradient purpose. A core cache hit "
                "does not remove current S/projection or imply a previous Hessian is reused.")
            checks["action_cache_enabled"] = _enable_updates
            checks["optimized_direct_control"] = _optimized_direct_control
            checks["action_diagnostics_measured"] = bool(validate and not _optimized_direct_control)
            yield phases, counts, checks, actual_oracle
    finally:
        protocol.DenseOracle, protocol.make_hooks = previous_oracle, previous_hooks


def run_one(problem, case, seed, T, **kwargs):
    global _enable_updates, _optimized_direct_control
    previous, previous_control = _enable_updates, _optimized_direct_control
    control = bool(case.get("optimized_direct_control", False))
    enabled = bool(case.get("action_cache_enabled", not control))
    if control and (T != 1 or enabled):
        raise ValueError("optimized_direct_control requires T1 and action_cache_enabled=false")
    _enable_updates, _optimized_direct_control = enabled, control
    try:
        result = _old_run_one(problem, case, seed, T, **kwargs)
        result["action_cache_enabled"] = _enable_updates
        result["optimized_direct_control"] = control
        if control:
            result["core"] = "dense_large_block_current"
            result["action_cache_variant"] = "optimized_direct_control_original_large_block"
        else:
            result["action_cache_variant"] = "fixed_A_linear_updates" if _enable_updates else "direct_A_products_same_oracle"
        return result
    finally:
        _enable_updates = previous
        _optimized_direct_control = previous_control


protocol.source_map = source_map
protocol.provenance = provenance
protocol.instrument_dense = instrument_dense
protocol.run_one = run_one
protocol.CORE = CORE

if __name__ == "__main__":
    protocol.main()
