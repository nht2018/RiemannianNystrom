import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from rsncn import (  # noqa: E402
    FullCubicOptions,
    NewtonOptions,
    SolverOptions,
    VanillaNewtonOptions,
    solve_spd_full_cubic,
    solve_spd_nystrom_cubic,
    solve_spd_nystrom_newton,
    solve_spd_vanilla_newton,
)

from PIL import Image
import math
from io import BytesIO

DEFAULT_CONFIG = "compare_spd_config.json"
PLOT_FONT_SIZE = 16
PLOT_TICK_SIZE = 14
PLOT_LEGEND_SIZE = 14
PLOT_SHOW_GRID = False


def load_config(default_filename: str, args: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--config", default=None, help="Path to JSON configuration file")
    parser.add_argument("-h", "--help", action="store_true", help="Show help message and exit")
    parsed, remaining = parser.parse_known_args(args)
    if parsed.help:
        parser.print_help()
        sys.exit(0)

    config_path = parsed.config or os.path.join(THIS_DIR, default_filename)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f), remaining
    if parsed.config:
        raise FileNotFoundError(f"Config file '{config_path}' not found")
    return {}, remaining


CONFIG, _ = load_config(DEFAULT_CONFIG, sys.argv[1:])

PLOT_CONFIG = CONFIG.get("plots", {})
PLOT_X_LIMITS = PLOT_CONFIG.get("x_limits", {})
PLOT_Y_LIMITS = PLOT_CONFIG.get("y_limits", {})
PLOT_LOG_Y = PLOT_CONFIG.get("log_y", {})
PROBLEM_DIM = int(CONFIG.get("problem", {}).get("d", 0) or 0)
TANGENT_DIM = PROBLEM_DIM * (PROBLEM_DIM + 1) // 2 if PROBLEM_DIM > 0 else None


def _format_display_label(label: str) -> str:
    """Prettify legend labels by removing redundant words and switching sketch notation."""
    cleaned = label.strip()
    sketch_match = re.search(r"sketch=(\d+)", cleaned)
    sketch = f"($\\ell$={sketch_match.group(1)})" if sketch_match else ""
    transport_match = re.search(r"T=(\d+|inf)", cleaned)
    transport = ""
    if transport_match:
        transport_val = transport_match.group(1)
        transport_label = r"\infty" if transport_val == "inf" else transport_val
        transport = f", T={transport_label}"
    switch_match = re.search(r"switch=([0-9eE+.\-]+)", cleaned)
    switch = ""
    if switch_match:
        switch = f", g={switch_match.group(1)}"
    if re.search(r"\bFull\b", cleaned, re.IGNORECASE):
        return "cubic Newton"
    if re.search(r"Nystr[oö]m|Nystrom", cleaned, re.IGNORECASE):
        base = "Nyström"
        if sketch:
            return f"{base} {sketch[:-1]}{transport}{switch})".strip()
        return f"{base}{transport}{switch}".strip()
    cleaned = cleaned.replace("Newton", "")
    cleaned = cleaned.replace("Cubic", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("( ", "(").replace(" )", ")")
    cleaned = cleaned.replace("sketch=", r"$\ell$=")
    return cleaned


def make_spd_matrix(dim: int, rng: np.random.Generator, eig_range: Tuple[float, float]) -> np.ndarray:
    low, high = eig_range
    low, high = float(min(low, high)), float(max(low, high))
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    vals = rng.uniform(low, high, size=dim)
    return (Q * vals) @ Q.T


def make_initial_spd(dim: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.standard_normal((dim, dim))
    X = M @ M.T
    w, Q = np.linalg.eigh(X)
    w = np.maximum(w, 1e-2)
    return (Q * w) @ Q.T


def write_history_csv(path: str, hist: Dict[str, List[float]]) -> None:
    base_keys = ["iter", "f", "grad_norm"]
    extra_cols = sorted(k for k in hist.keys() if k not in {"f", "grad_norm"})
    columns = base_keys + extra_cols
    rows = max(len(hist["f"]), len(hist["grad_norm"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for it in range(rows):
            row: Dict[str, Any] = {
                "iter": it,
                "f": hist["f"][it] if it < len(hist["f"]) else hist["f"][-1],
                "grad_norm": hist["grad_norm"][it] if it < len(hist["grad_norm"]) else hist["grad_norm"][-1],
            }
            for key in extra_cols:
                series = hist.get(key, [])
                if not series:
                    continue
                if it < len(series):
                    row[key] = series[it]
                else:
                    row[key] = series[-1]
            writer.writerow(row)


def _plot_key(title: str) -> str:
    return title.strip().lower().replace(" ", "_")


def _configured_y_limits(title: str, x_key: str, y_key: str) -> Optional[Tuple[float, float]]:
    for key in (_plot_key(title), f"{y_key}_vs_{x_key}", y_key):
        limits = PLOT_Y_LIMITS.get(key)
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            return float(limits[0]), float(limits[1])
    return None


def _configured_x_limits(title: str, x_key: str, y_key: str) -> Optional[Any]:
    for key in (_plot_key(title), f"{y_key}_vs_{x_key}", x_key):
        limits = PLOT_X_LIMITS.get(key)
        if isinstance(limits, str):
            return limits
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            return float(limits[0]), float(limits[1])
    return None


def _configured_log_y(title: str, x_key: str, y_key: str, default: bool) -> bool:
    for key in (_plot_key(title), f"{y_key}_vs_{x_key}", y_key):
        value = PLOT_LOG_Y.get(key)
        if isinstance(value, bool):
            return value
    return default


def load_history_csv(path: str) -> Dict[str, List[float]]:
    hist: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key == "iter" or val == "":
                    continue
                hist.setdefault(key, []).append(float(val))
    if "f" not in hist or "grad_norm" not in hist:
        raise ValueError("history CSV missing 'f' or 'grad_norm'")
    return hist


def combine_images(images: List[Image.Image], output_path: str, cols: int = 2) -> None:
    if not images:
        return
    widths, heights = zip(*(img.size for img in images))
    max_w = max(widths)
    max_h = max(heights)
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * max_w, rows * max_h), color="white")
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        canvas.paste(img, (c * max_w, r * max_h))
    canvas.save(output_path)


def plot_histories(case_name: str, histories: List[Tuple[str, Dict[str, List[float]], int]], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable; skipping plots ({exc})")
        return

    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_TICK_SIZE,
            "ytick.labelsize": PLOT_TICK_SIZE,
            "legend.fontsize": PLOT_LEGEND_SIZE,
        }
    )

    panels: List[Image.Image] = []
    panel_buffers: List[BytesIO] = []

    def extract_sketch_size(label: str) -> Optional[int]:
        match = re.search(r"sketch=(\d+)", label)
        if match:
            return int(match.group(1))
        return None

    sketch_entries: List[Tuple[int, str]] = []
    for label, _, _ in histories:
        sketch_size = extract_sketch_size(label)
        if sketch_size is not None:
            sketch_entries.append((sketch_size, label))
    display_labels: Dict[str, str] = {label: _format_display_label(label) for label, _, _ in histories}

    for _, hist, _ in histories:
        if "hess_nystrom_gaussian_bound_rel" in hist:
            continue
        bound_op = np.asarray(hist.get("hess_nystrom_gaussian_bound_op", []), dtype=float)
        err_op = np.asarray(hist.get("hess_nystrom_err_op", []), dtype=float)
        err_rel = np.asarray(hist.get("hess_nystrom_err_rel", []), dtype=float)
        if bound_op.size == 0 or err_op.size == 0 or err_rel.size == 0:
            continue
        n = min(bound_op.size, err_op.size, err_rel.size)
        rel_bound = np.full(n, np.nan, dtype=float)
        valid = np.isfinite(bound_op[:n]) & np.isfinite(err_op[:n]) & np.isfinite(err_rel[:n]) & (np.abs(err_op[:n]) > 0.0)
        rel_bound[valid] = bound_op[:n][valid] * err_rel[:n][valid] / err_op[:n][valid]
        hist["hess_nystrom_gaussian_bound_rel"] = rel_bound.tolist()

    sketch_entries.sort(key=lambda item: item[0])
    red_palette = ["#f6c7c4", "#f8b4b4", "#f17c7c", "#e24c4b", "#c81d25", "#8c1c13"]
    sketch_styles: Dict[str, Dict[str, Any]] = {}
    n_sketch = len(sketch_entries)
    palette_positions = (
        np.linspace(0, len(red_palette) - 1, num=n_sketch)
        if n_sketch
        else []
    )
    for idx, (_, label) in enumerate(sketch_entries):
        palette_idx = int(round(palette_positions[idx]))
        palette_idx = max(0, min(palette_idx, len(red_palette) - 1))
        color = red_palette[palette_idx]
        sketch_styles[label] = {
            "color": color,
            "palette_rank": palette_idx,
            "linestyle": "--" if "transport" in label else "-",
        }

    green_palette = ["#c7e9c0", "#74c476", "#238b45"]
    bound_styles: Dict[str, Dict[str, Any]] = {}
    for idx, (_, label) in enumerate(sketch_entries):
        bound_styles[label] = {
            "color": green_palette[min(idx, len(green_palette) - 1)],
            "linestyle": "--",
        }

    def render_plot(
        x_key: str,
        y_key: str,
        title: str,
        xlabel: str,
        ylabel: str,
        logy: bool = False,
        bound_y_key: Optional[str] = None,
        bound_label_prefix: str = "Gaussian bound",
    ) -> None:
        logy = _configured_log_y(title, x_key, y_key, logy)
        fig, ax = plt.subplots(figsize=(6, 4))

        def apply_legend() -> None:
            if bound_y_key is not None:
                from matplotlib.legend_handler import HandlerTuple

                legend_handles = []
                legend_labels = []
                seen_sizes = []
                for sketch_size, label in sketch_entries:
                    if sketch_size in seen_sizes:
                        continue
                    seen_sizes.append(sketch_size)
                    main_style = sketch_styles.get(label, {"color": "#c81d25", "linestyle": "-", "linewidth": 2.0})
                    bound_style = bound_styles.get(label, {"color": "#238b45", "linestyle": "--", "linewidth": 2.0})
                    main_handle = plt.Line2D([0], [0], color=main_style.get("color", "#c81d25"), linestyle="-", linewidth=main_style.get("linewidth", 2.0))
                    bound_handle = plt.Line2D([0], [0], color=bound_style.get("color", "#238b45"), linestyle="--", linewidth=bound_style.get("linewidth", 2.0))
                    legend_handles.append((main_handle, bound_handle))
                    legend_labels.append(rf"$\ell = {sketch_size}$")
                if legend_handles:
                    ax.legend(
                        legend_handles,
                        legend_labels,
                        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.8)},
                        loc="upper right",
                        ncol=1,
                        frameon=False,
                        handlelength=4.0,
                        borderaxespad=0.4,
                    )
                return
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, frameon=False)

        eps = np.finfo(float).tiny
        series: List[Dict[str, Any]] = []
        finite_min = np.inf
        finite_max = -np.inf
        iter_start = 1 if x_key == "iter" and y_key.startswith("hess_") else 0
        shared_min_final_time: Optional[float] = None

        for label, hist, _ in histories:
            yarr = np.asarray(hist.get(y_key, []), dtype=float)
            if yarr.size:
                finite_vals = yarr[np.isfinite(yarr)]
                if finite_vals.size:
                    finite_min = min(finite_min, float(np.min(finite_vals)))
                    finite_max = max(finite_max, float(np.max(finite_vals)))
                if x_key == "iter":
                    xvals = np.arange(iter_start, iter_start + len(yarr))
                else:
                    xvals = np.asarray(hist.get(x_key, list(range(len(yarr)))), dtype=float)
                series.append(
                    {
                        "base_label": label,
                        "plot_label": display_labels.get(label, label),
                        "yarr": yarr,
                        "xvals": xvals,
                        "kind": "main",
                    }
                )
                if x_key == "time":
                    finite_x = xvals[np.isfinite(xvals)]
                    if finite_x.size:
                        final_time = float(finite_x[-1])
                        if shared_min_final_time is None:
                            shared_min_final_time = final_time
                        else:
                            shared_min_final_time = min(shared_min_final_time, final_time)

            if bound_y_key is not None:
                bound_arr = np.asarray(hist.get(bound_y_key, []), dtype=float)
                if bound_arr.size:
                    finite_vals = bound_arr[np.isfinite(bound_arr)]
                    if finite_vals.size:
                        finite_min = min(finite_min, float(np.min(finite_vals)))
                        finite_max = max(finite_max, float(np.max(finite_vals)))
                    if x_key == "iter":
                        bound_xvals = np.arange(iter_start, iter_start + len(bound_arr))
                    else:
                        bound_xvals = np.asarray(hist.get(x_key, list(range(len(bound_arr)))), dtype=float)
                    sketch_size = extract_sketch_size(label)
                    if sketch_size is not None:
                        plot_label = f"{bound_label_prefix} ($\\ell$={sketch_size})"
                    else:
                        plot_label = bound_label_prefix
                    series.append(
                        {
                            "base_label": label,
                            "plot_label": plot_label,
                            "yarr": bound_arr,
                            "xvals": bound_xvals,
                            "kind": "bound",
                        }
                    )

        def series_key(item: Dict[str, Any]) -> Tuple[int, int, int]:
            label = item["base_label"]
            kind_rank = 0 if item["kind"] == "main" else 1
            if re.search(r"\bFull\b", label, re.IGNORECASE):
                return (0, 0, kind_rank)
            if re.search(r"Nystr[oö]m|Nystrom", label, re.IGNORECASE):
                sketch_size = extract_sketch_size(label)
                return (1, sketch_size if sketch_size is not None else 0, kind_rank)
            return (2, 0, kind_rank)

        ordered_series = sorted(series, key=series_key)

        collected_linear: List[float] = []
        collected_positive: List[float] = []
        shift = 0.0
        if logy:
            lower_limit = 1e-15
            if not series:
                return
            if not (np.isfinite(finite_min) and np.isfinite(finite_max)):
                shift = 1e-14
            elif finite_min <= 0:
                shift = max(1e-14, -finite_min + 1e-14)
            else:
                shift = 0.0

            def build_plots(current_shift: float) -> Tuple[List[float], List[Dict[str, Any]]]:
                positives: List[float] = []
                plots: List[Dict[str, Any]] = []
                for item in ordered_series:
                    yarr = item["yarr"]
                    xvals = item["xvals"]
                    finite_mask = np.isfinite(yarr)
                    y_shifted = yarr + current_shift
                    y_shifted = np.where(finite_mask, y_shifted, np.nan)
                    pos = y_shifted[(finite_mask) & (y_shifted > eps)]
                    if pos.size:
                        positives.extend(pos.tolist())
                    y_plot = np.where(finite_mask, np.clip(y_shifted, eps, None), np.nan)
                    plot_item = item.copy()
                    plot_item["y_plot"] = y_plot
                    plots.append(plot_item)
                return positives, plots

            positives, plots = build_plots(shift)
            if positives:
                min_pos = min(positives)
                if min_pos < lower_limit:
                    shift += lower_limit - min_pos + lower_limit * 0.1
                    positives, plots = build_plots(shift)
            else:
                shift += lower_limit
            positives, plots = build_plots(shift)

            for item in plots:
                xvals = item["xvals"]
                y_plot = item["y_plot"]
                plot_kwargs: Dict[str, Any] = {"label": item["plot_label"]}
                if item["kind"] == "bound":
                    plot_kwargs.update(bound_styles.get(item["base_label"], {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.0}))
                else:
                    style = sketch_styles.get(item["base_label"])
                    if style:
                        plot_kwargs.update({k: v for k, v in style.items() if k != "palette_rank"})
                        if xvals.size:
                            plot_kwargs.setdefault("markevery", max(1, xvals.size // 20 or 1))
                ax.plot(xvals, y_plot, **plot_kwargs)

            ax.set_xlabel(xlabel)
            x_limits = _configured_x_limits(title, x_key, y_key)
            if isinstance(x_limits, str):
                if x_limits == "shared_min_final" and shared_min_final_time is not None:
                    ax.set_xlim(0.0, shared_min_final_time)
            elif x_limits is not None:
                ax.set_xlim(*x_limits)
            label_shift = f" + {shift:.2e}" if shift > 0 else ""
            ax.set_ylabel(f"{ylabel}{label_shift}")
            ax.set_yscale("log")
            if positives:
                ymin = min(positives)
                ymax = max(positives)
                if ymin <= 0:
                    ymin = eps
                log_min = np.log10(ymin)
                log_max = np.log10(max(ymax, ymin * 1.01))
                pad = max(0.1 * (log_max - log_min) if log_max > log_min else 0.25, 0.25)
                lower = 10 ** (log_min - pad)
                upper = 10 ** (log_max + pad)
                ax.set_ylim(max(lower, eps), upper)
            else:
                ax.set_ylim(lower_limit, 1.0)
            y_limits = _configured_y_limits(title, x_key, y_key)
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if PLOT_SHOW_GRID:
                ax.grid(True, alpha=0.3)
            apply_legend()
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            panels.append(img)
            panel_buffers.append(buf)
            png_path = os.path.join(out_dir, f"{title.replace(' ', '_').lower()}_{case_name}.png")
            fig.savefig(png_path, dpi=150)
            eps_path = png_path.replace(".png", ".eps")
            fig.savefig(eps_path, format="eps")
            plt.close(fig)
            print(f"[saved] {png_path}")
            return

        for item in ordered_series:
            yarr = item["yarr"]
            xvals = item["xvals"]
            finite_mask = np.isfinite(yarr)
            if logy:
                y_shifted = yarr + shift
                y_shifted = np.where(finite_mask, y_shifted, np.nan)
                positive_vals = y_shifted[(finite_mask) & (y_shifted > eps)]
                if positive_vals.size:
                    collected_positive.extend(positive_vals.tolist())
                y_plot = np.where(finite_mask, np.clip(y_shifted, eps, None), np.nan)
            else:
                finite_vals = yarr[finite_mask]
                if finite_vals.size:
                    collected_linear.extend(finite_vals.tolist())
                y_plot = np.where(finite_mask, yarr, np.nan)
            plot_kwargs = {"label": item["plot_label"]}
            if item["kind"] == "bound":
                plot_kwargs.update(bound_styles.get(item["base_label"], {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.0}))
            else:
                style = sketch_styles.get(item["base_label"])
                if style:
                    plot_kwargs.update({k: v for k, v in style.items() if k != "palette_rank"})
                    if xvals.size:
                        plot_kwargs.setdefault("markevery", max(1, len(xvals) // 20 or 1))
            ax.plot(xvals, y_plot, **plot_kwargs)
        ax.set_xlabel(xlabel)
        x_limits = _configured_x_limits(title, x_key, y_key)
        if isinstance(x_limits, str):
            if x_limits == "shared_min_final" and shared_min_final_time is not None:
                ax.set_xlim(0.0, shared_min_final_time)
        elif x_limits is not None:
            ax.set_xlim(*x_limits)
        if logy and shift > 0.0:
            ax.set_ylabel(f"{ylabel} + {shift:.2e}")
        else:
            ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
            if collected_positive:
                ymin = min(collected_positive)
                ymax = max(collected_positive)
                if ymin <= 0:
                    ymin = eps
                log_min = np.log10(ymin)
                log_max = np.log10(max(ymax, ymin * 1.01))
                pad = max(0.1 * (log_max - log_min) if log_max > log_min else 0.25, 0.25)
                lower = 10 ** (log_min - pad)
                upper = 10 ** (log_max + pad)
                ax.set_ylim(max(lower, eps), upper)
        elif collected_linear:
            ymin = min(collected_linear)
            ymax = max(collected_linear)
            if ymin == ymax:
                delta = abs(ymin) * 0.1 or 1.0
                ax.set_ylim(ymin - delta, ymax + delta)
            else:
                margin = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - margin, ymax + margin)
        y_limits = _configured_y_limits(title, x_key, y_key)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if PLOT_SHOW_GRID:
            ax.grid(True, alpha=0.3)
        apply_legend()
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        panels.append(img)
        panel_buffers.append(buf)
        png_path = os.path.join(out_dir, f"{title.replace(' ', '_').lower()}_{case_name}.png")
        fig.savefig(png_path, dpi=150)
        eps_path = png_path.replace(".png", ".eps")
        fig.savefig(eps_path, format="eps")
        plt.close(fig)
        print(f"[saved] {png_path}")

    render_plot("time", "grad_norm", "Gradient norm vs time", "time [s]", "||grad||", logy=True)
    render_plot("time", "f", "Objective vs time", "time [s]", "f", logy=True)
    render_plot("iter", "grad_norm", "Gradient norm vs iteration", "iteration", "||grad||", logy=True)
    render_plot("iter", "f", "Objective vs iteration", "iteration", "f", logy=True)
    render_plot(
        "iter",
        "hess_nystrom_err_rel",
        "Nystrom Hessian relative error vs iteration",
        "iteration",
        r"$\Vert H - H_{\mathrm{nys}} \Vert_{\mathrm{op}} \,/\, \Vert H \Vert_{\mathrm{op}}$",
        logy=False,
        bound_y_key="hess_nystrom_gaussian_bound_rel",
        bound_label_prefix="bound",
    )
    render_plot(
        "iter",
        "hess_nystrom_err_op",
        "Nystrom Hessian error vs iteration",
        "iteration",
        r"$\Vert H - H_{\mathrm{nys}} \Vert_{\mathrm{op}}$",
        logy=True,
        bound_y_key="hess_nystrom_gaussian_bound_op",
        bound_label_prefix="bound",
    )

    if panels:
        output = os.path.join(out_dir, f"compare_spd_panels_{case_name}.png")
        combine_images(panels, output)
        print(f"[saved] {output}")
    for img, buf in zip(panels, panel_buffers):
        img.close()
        buf.close()


DEFAULT_PROBLEM = {
    "d": 10,
    "seed": 0,
    "init_seeds": [0],
    "eig_min": 0.5,
    "eig_max": 2.0,
    "dist_weight": 1.0,
    "inv_linear_weight": 1.0,
    "inv_quadratic_weight": 1.0,
}

PROBLEM_CONFIG = DEFAULT_PROBLEM.copy()
PROBLEM_CONFIG.update(CONFIG.get("problem", {}))

METHOD_CONFIGS = {
    "vanilla": {
        "label": "Vanilla Newton",
        "short": "vanilla",
        "solver": solve_spd_vanilla_newton,
        "option_class": VanillaNewtonOptions,
        "base_options": {
            "max_iter": 100000,
            "tol_grad": 1e-14,
            "ls_c1": 1e-4,
            "ls_beta": 0.5,
            "ls_max_steps": 20,
            "ridge": 1e-12,
            "display_progress": False,
            "time_limit": 50.0,
        },
    },
    "nystrom": {
        "label": "Nyström Newton",
        "short": "nystrom",
        "solver": solve_spd_nystrom_newton,
        "option_class": NewtonOptions,
        "base_options": {
            "max_iter": 100000,
            "tol_grad": 1e-14,
            "sketch_size": 20,
            "ls_c1": 1e-4,
            "ls_beta": 0.5,
            "ls_max_steps": 20,
            "ridge": 1e-12,
            "display_progress": False,
            "time_limit": 50.0,
        },
    },
    "full_cubic": {
        "label": "Full Cubic Newton",
        "short": "full_cubic",
        "solver": solve_spd_full_cubic,
        "option_class": FullCubicOptions,
        "base_options": {
            "max_iter": 100000,
            "tol_grad": 1e-14,
            "tol_pred": 1e-10,
            "tol_step": 1e-14,
            "sigma0": 2.0,
            "sigma_max": None,
            "eta1": 0.1,
            "eta2": 0.9,
            "gamma1": 0.5,
            "gamma2": 2.0,
            "ridge": 1e-12,
            "display_progress": False,
            "time_limit": 50.0,
        },
    },
    "nystrom_cubic": {
        "label": "Nyström Cubic Newton",
        "short": "nystrom_cubic",
        "solver": solve_spd_nystrom_cubic,
        "option_class": SolverOptions,
        "base_options": {
            "max_iter": 100000,
            "tol_grad": 1e-14,
            "tol_pred": 1e-10,
            "tol_step": 1e-14,
            "sigma0": 2.0,
            "sigma_max": None,
            "sketch_size": 15,
            "eta1": 0.1,
            "eta2": 0.9,
            "gamma1": 0.5,
            "gamma2": 2.0,
            "ridge": 1e-12,
            "display_progress": False,
            "time_limit": 50.0,
        },
    },
}

def _make_nystrom_cubic_variant(sketch_size: int) -> Dict[str, Any]:
    """Return a METHOD_CONFIGS entry for a Nyström cubic solver variant."""
    template = METHOD_CONFIGS["nystrom_cubic"]
    base_opts = template["base_options"].copy()
    base_opts["sketch_size"] = sketch_size
    return {
        "label": f"Nyström Cubic Newton (sketch={sketch_size})",
        "short": f"nystrom_cubic_s{sketch_size}",
        "solver": template["solver"],
        "option_class": template["option_class"],
        "base_options": base_opts,
    }


def _make_nystrom_cubic_transport_variant(
    sketch_size: int,
    refresh_every: Optional[int],
    switch_grad_tol: Optional[float] = None,
) -> Dict[str, Any]:
    template = METHOD_CONFIGS["nystrom_cubic"]
    base_opts = template["base_options"].copy()
    base_opts["sketch_size"] = sketch_size
    base_opts["transport_sketch"] = True
    base_opts["transport_refresh_every"] = 0 if refresh_every is None else refresh_every
    if switch_grad_tol is not None:
        base_opts["transport_disable_grad_tol"] = switch_grad_tol
    refresh_suffix = "inf" if refresh_every is None else str(refresh_every)
    refresh_label = r"\infty" if refresh_every is None else str(refresh_every)
    switch_suffix = ""
    switch_label = ""
    if switch_grad_tol is not None:
        switch_text = f"{switch_grad_tol:.0e}"
        switch_suffix = f", switch={switch_text}"
        switch_label = f", g={switch_text}"
    return {
        "label": f"Nyström Cubic Newton (sketch={sketch_size}, transport, T={refresh_suffix}{switch_suffix})",
        "short": f"nystrom_cubic_s{sketch_size}_transport_t{refresh_suffix}" + (f"_switch_g{switch_grad_tol:.0e}" if switch_grad_tol is not None else ""),
        "solver": template["solver"],
        "option_class": template["option_class"],
        "base_options": base_opts,
        "display_label": f"Nyström ($\\ell$={sketch_size}, T={refresh_label}{switch_label})",
    }


def _ensure_nystrom_cubic_variant(sketch_size: int) -> None:
    """Ensure a Nyström cubic variant for ``sketch_size`` exists."""
    key = f"nystrom_cubic_s{sketch_size}"
    if key in METHOD_CONFIGS:
        METHOD_CONFIGS[key]["base_options"]["sketch_size"] = sketch_size
        return
    METHOD_CONFIGS[key] = _make_nystrom_cubic_variant(sketch_size)


def _ensure_nystrom_cubic_transport_variant(
    sketch_size: int,
    refresh_every: Optional[int],
    switch_grad_tol: Optional[float] = None,
) -> None:
    refresh_suffix = "inf" if refresh_every is None else str(refresh_every)
    key = f"nystrom_cubic_s{sketch_size}_transport_t{refresh_suffix}"
    if switch_grad_tol is not None:
        key = f"{key}_switch_g{switch_grad_tol:.0e}"
    if key in METHOD_CONFIGS:
        METHOD_CONFIGS[key]["base_options"]["sketch_size"] = sketch_size
        METHOD_CONFIGS[key]["base_options"]["transport_sketch"] = True
        METHOD_CONFIGS[key]["base_options"]["transport_refresh_every"] = 0 if refresh_every is None else refresh_every
        if switch_grad_tol is not None:
            METHOD_CONFIGS[key]["base_options"]["transport_disable_grad_tol"] = switch_grad_tol
        return
    METHOD_CONFIGS[key] = _make_nystrom_cubic_transport_variant(sketch_size, refresh_every, switch_grad_tol)


def _inject_config_variants(config: Dict[str, Any]) -> None:
    """Add any Nyström cubic variants referenced in the config."""
    method_keys = set(config.get("methods", {}).keys())
    compare_keys = set(config.get("compare_methods", []))
    for key in method_keys | compare_keys:
        if not key.startswith("nystrom_cubic_s"):
            continue
        switch_match = re.fullmatch(r"nystrom_cubic_s(\d+)_transport_t(\d+|inf)_switch_g([0-9eE+.\-]+)", key)
        if switch_match:
            sketch_size = int(switch_match.group(1))
            refresh_token = switch_match.group(2)
            refresh_every = None if refresh_token == "inf" else int(refresh_token)
            switch_grad_tol = float(switch_match.group(3))
            _ensure_nystrom_cubic_transport_variant(sketch_size, refresh_every, switch_grad_tol)
            continue
        transport_match = re.fullmatch(r"nystrom_cubic_s(\d+)_transport_t(\d+|inf)", key)
        if transport_match:
            sketch_size = int(transport_match.group(1))
            refresh_token = transport_match.group(2)
            refresh_every = None if refresh_token == "inf" else int(refresh_token)
            _ensure_nystrom_cubic_transport_variant(sketch_size, refresh_every)
            continue
        plain_match = re.fullmatch(r"nystrom_cubic_s(\d+)", key)
        if plain_match:
            sketch_size = int(plain_match.group(1))
            _ensure_nystrom_cubic_variant(sketch_size)
            continue


_inject_config_variants(CONFIG)


COMPARE_METHODS = CONFIG.get(
    "compare_methods",
    ["vanilla", "nystrom", "full_cubic", "nystrom_cubic"],
)


def main() -> None:
    d = int(PROBLEM_CONFIG["d"])
    seed = int(PROBLEM_CONFIG.get("seed", 0))
    init_seeds = list(PROBLEM_CONFIG.get("init_seeds", [0]))
    eig_min = float(PROBLEM_CONFIG.get("eig_min", 0.5))
    eig_max = float(PROBLEM_CONFIG.get("eig_max", 2.0))
    dist_weight = float(PROBLEM_CONFIG.get("dist_weight", 1.0))
    inv_linear_weight = float(PROBLEM_CONFIG.get("inv_linear_weight", 1.0))
    inv_quadratic_weight = float(PROBLEM_CONFIG.get("inv_quadratic_weight", 1.0))

    rng_global = np.random.default_rng(seed)
    A = make_spd_matrix(d, rng_global, (eig_min, eig_max))
    B = make_spd_matrix(d, rng_global, (eig_min, eig_max))

    out_root = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "reports-spd"))
    os.makedirs(out_root, exist_ok=True)
    run_name = CONFIG.get("run_name", f"spd_d{d}_seed{seed}")
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    method_overrides = CONFIG.get("methods", {})
    summary_rows: List[Dict[str, str]] = []
    has_hesserr = False

    for init_seed in init_seeds:
        rng = np.random.default_rng(10_000 + init_seed)
        X0 = make_initial_spd(d, rng)

        histories_for_plot: List[Tuple[str, Dict[str, List[float]], int]] = []

        for method_name in COMPARE_METHODS:
            spec = METHOD_CONFIGS[method_name]
            overrides = method_overrides.get(method_name, {})
            option_kwargs = spec["base_options"].copy()
            option_kwargs.update(overrides.get("options", {}))
            fields = getattr(spec["option_class"], "__dataclass_fields__", {})
            if "random_state" in fields:
                option_kwargs.setdefault("random_state", init_seed)
            if "dist_weight" in fields:
                option_kwargs.setdefault("dist_weight", dist_weight)
            if "inv_linear_weight" in fields:
                option_kwargs.setdefault("inv_linear_weight", inv_linear_weight)
            if "inv_quadratic_weight" in fields:
                option_kwargs.setdefault("inv_quadratic_weight", inv_quadratic_weight)
            opts = spec["option_class"](**option_kwargs)
            if getattr(opts, "track_hess_nystrom_error", False):
                has_hesserr = True

            csv_tag = f"{spec['short']}_d{d}_init{init_seed}"
            if getattr(opts, "track_hess_nystrom_error", False):
                csv_tag = f"{csv_tag}_hesserr"
            csv_path = os.path.join(out_dir, f"{csv_tag}.csv")

            rerun = overrides.get("rerun", True)
            if rerun or not os.path.exists(csv_path):
                X_sol, hist = spec["solver"](A, B, X0, opts)
                write_history_csv(csv_path, hist)
                print(f"[saved] {csv_path}")
            else:
                hist = load_history_csv(csv_path)

            final_f = hist["f"][-1]
            final_grad = hist["grad_norm"][-1]
            time_series = hist.get("time", [])
            final_time = time_series[-1] if time_series else float("nan")
            iterations = len(hist["f"]) - 1
            accepted = hist.get("accepted", [])
            accept_rate = np.mean(accepted) if accepted else float("nan")

            summary_rows.append(
                {
                    "method": spec["label"],
                    "init_seed": str(init_seed),
                    "iterations": str(iterations),
                    "final_f": f"{final_f:.6e}",
                    "final_grad": f"{final_grad:.3e}",
                    "time_s": f"{final_time:.3f}" if not np.isnan(final_time) else "",
                    "accept_rate": f"{accept_rate:.2f}" if not np.isnan(accept_rate) else "",
                    "csv": os.path.relpath(csv_path, start=out_root),
                }
            )

            histories_for_plot.append((spec["label"], hist, getattr(opts, "max_iter", iterations)))

        case_name = f"{run_name}_init{init_seed}"
        if any(hist.get("hess_nystrom_err_op") for _, hist, _ in histories_for_plot):
            case_name = f"{case_name}_hesserr"
        plot_histories(case_name, histories_for_plot, out_dir)

    summary_csv_name = "compare_spd_summary_hesserr.csv" if has_hesserr else "compare_spd_summary.csv"
    summary_csv = os.path.join(out_dir, summary_csv_name)
    headers = ["method", "init_seed", "iterations", "final_f", "final_grad", "time_s", "accept_rate", "csv"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[saved] {summary_csv}")


if __name__ == "__main__":
    main()
