import argparse
import csv
import json
import math
import os
import re
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from rsncn import SolverOptions, solve_grassmann_nystrom_cubic  # noqa: E402

DEFAULT_CONFIG = "compare_grassmann_transport_demo.json"
PLOT_ONLY = os.environ.get("GRASSMANN_PLOT_ONLY", "").strip().lower() in {"1", "true", "yes", "y"}
PLOT_FONT_SIZE = 16
PLOT_TICK_SIZE = 14
PLOT_LEGEND_SIZE = 14


def build_spectrum(problem: Dict[str, Any]) -> np.ndarray:
    n = int(problem.get("n", 2000))
    p = int(problem.get("p", 20))
    if "spectrum_values" in problem:
        values = np.asarray(problem["spectrum_values"], dtype=float)
        if values.shape != (n,):
            raise ValueError("spectrum_values must have length n.")
        return values

    spectrum_top = float(problem.get("spectrum_top", 20.0))
    spectrum_bottom = float(problem.get("spectrum_bottom", 1.0))
    spectrum_mode = str(problem.get("spectrum_mode", "linear")).lower()
    if spectrum_mode != "clustered":
        return np.linspace(spectrum_top, spectrum_bottom, n)

    cluster_size = int(problem.get("cluster_size", max(4 * p, 40)))
    cluster_size = max(1, min(cluster_size, n))
    cluster_bottom = float(problem.get("cluster_bottom", spectrum_top - 0.5))
    tail_top = float(problem.get("tail_top", cluster_bottom - 0.02))

    head = np.linspace(spectrum_top, cluster_bottom, cluster_size, endpoint=True)
    if cluster_size == n:
        return head
    tail = np.linspace(tail_top, spectrum_bottom, n - cluster_size, endpoint=True)
    return np.concatenate([head, tail])


def load_config(default_filename: str, argv: List[str]) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-f", "--config", default=os.path.join(THIS_DIR, default_filename))
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config(DEFAULT_CONFIG, sys.argv[1:])
PLOT_CONFIG = CONFIG.get("plots", {})
PLOT_X_LIMITS = PLOT_CONFIG.get("x_limits", {})
PLOT_Y_LIMITS = PLOT_CONFIG.get("y_limits", {})
PLOT_LOG_Y = PLOT_CONFIG.get("log_y", {})
PLOT_SOLID_LINES = bool(PLOT_CONFIG.get("solid_lines", False))
PLOT_LABEL_COLORS = PLOT_CONFIG.get("label_colors", {})
PLOT_LINEWIDTH = PLOT_CONFIG.get("linewidth", None)
PLOT_SHOW_GRID = bool(PLOT_CONFIG.get("show_grid", False))


def write_history_csv(path: str, hist: Dict[str, List[float]]) -> None:
    fieldnames = ["iter", "f", "grad_norm", "time_s", "sigma", "rho", "accepted", "ell"]
    count = len(hist["f"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(count):
            row = {
                "iter": i,
                "f": hist["f"][i],
                "grad_norm": hist["grad_norm"][i],
                "time_s": hist["time"][i],
                "sigma": hist["sigma"][i] if i < len(hist["sigma"]) else hist["sigma"][-1],
                "rho": hist["rho"][i - 1] if i > 0 and i - 1 < len(hist["rho"]) else "",
                "accepted": hist["accepted"][i - 1] if i > 0 and i - 1 < len(hist["accepted"]) else "",
                "ell": hist["ell"][i - 1] if i > 0 and i - 1 < len(hist["ell"]) else "",
            }
            writer.writerow(row)


def load_history_csv(path: str) -> Dict[str, List[float]]:
    hist: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key == "iter" or val == "":
                    continue
                hist.setdefault("time" if key == "time_s" else key, []).append(float(val))
    if "f" not in hist or "grad_norm" not in hist:
        raise ValueError(f"history CSV missing required columns: {path}")
    return hist


def _format_label(method_key: str, method_cfg: Dict[str, Any], opts: Dict[str, Any]) -> str:
    ell = int(opts.get("sketch_size", 0))
    if method_cfg.get("transport", False):
        T = method_cfg.get("transport_refresh_every", 0)
        return f"Nyström ($\\ell$={ell}, T={T})"
    return f"Nyström ($\\ell$={ell})"


def _plot_key(title: str) -> str:
    return title.strip().lower().replace(" ", "_")


def _configured_x_limits(title: str, x_key: str, y_key: str) -> Optional[Any]:
    for key in (_plot_key(title), f"{y_key}_vs_{x_key}", x_key):
        limits = PLOT_X_LIMITS.get(key)
        if isinstance(limits, str):
            return limits
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            return float(limits[0]), float(limits[1])
    return None


def _configured_y_limits(title: str, x_key: str, y_key: str) -> Optional[Tuple[float, float]]:
    for key in (_plot_key(title), f"{y_key}_vs_{x_key}", y_key):
        limits = PLOT_Y_LIMITS.get(key)
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            return float(limits[0]), float(limits[1])
    return None


def _configured_log_y(title: str, x_key: str, y_key: str, default: bool) -> bool:
    for key in (_plot_key(title), f"{y_key}_vs_{x_key}", y_key):
        value = PLOT_LOG_Y.get(key)
        if isinstance(value, bool):
            return value
    return default


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


def _extract_sketch_size(label: str) -> Optional[int]:
    match = re.search(r"\\ell\$=(\d+)", label)
    if match:
        return int(match.group(1))
    match = re.search(r"\((?:[^)]*?)=(\d+)", label)
    if match:
        return int(match.group(1))
    return None


def _extract_transport_period(label: str) -> Optional[int]:
    match = re.search(r"T=(\d+)", label)
    if match:
        return int(match.group(1))
    return None


def plot_histories(case_name: str, histories: List[Tuple[str, Dict[str, List[float]]]], out_dir: str) -> None:
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

    red_palette = ["#f6c7c4", "#f8b4b4", "#f17c7c", "#e24c4b", "#c81d25", "#8c1c13"]
    sorted_labels = sorted(
        [label for label, _ in histories],
        key=lambda label: (
            _extract_sketch_size(label) if _extract_sketch_size(label) is not None else 0,
            _extract_transport_period(label) if _extract_transport_period(label) is not None else -1,
        ),
    )
    style_map: Dict[str, Dict[str, Any]] = {}
    if sorted_labels:
        positions = np.linspace(0, len(red_palette) - 1, num=len(sorted_labels))
        for idx, label in enumerate(sorted_labels):
            palette_idx = int(round(float(positions[idx])))
            palette_idx = max(0, min(palette_idx, len(red_palette) - 1))
            style_map[label] = {
                "color": PLOT_LABEL_COLORS.get(label, red_palette[palette_idx]),
                "linestyle": "-" if PLOT_SOLID_LINES else ("--" if _extract_transport_period(label) is not None else "-"),
            }
            if PLOT_LINEWIDTH is not None:
                style_map[label]["linewidth"] = float(PLOT_LINEWIDTH)

    def render_plot(
        x_key: str,
        y_key: str,
        title: str,
        xlabel: str,
        ylabel: str,
        logy: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        logy = _configured_log_y(title, x_key, y_key, logy)

        shared_min_final_time: Optional[float] = None
        linear_vals: List[float] = []
        positive_vals: List[float] = []

        ordered = sorted(
            histories,
            key=lambda item: (
                _extract_sketch_size(item[0]) if _extract_sketch_size(item[0]) is not None else 0,
                _extract_transport_period(item[0]) if _extract_transport_period(item[0]) is not None else -1,
            ),
        )

        for label, hist in ordered:
            if x_key == "iter":
                xvals = np.arange(len(hist[y_key]), dtype=float)
            elif x_key == "time_s":
                # Internal history uses "time"; CSV uses "time_s".
                # Prefer the in-memory key to avoid silently falling back to iteration index.
                xvals = np.asarray(hist.get("time", hist.get("time_s", list(range(len(hist[y_key]))))), dtype=float)
            else:
                xvals = np.asarray(hist.get(x_key, list(range(len(hist[y_key])))), dtype=float)
            yvals = np.asarray(hist.get(y_key, []), dtype=float)
            finite_mask = np.isfinite(xvals) & np.isfinite(yvals)
            xvals = xvals[finite_mask]
            yvals = yvals[finite_mask]
            if not yvals.size:
                continue
            style = style_map.get(label, {"linewidth": 2.0})
            ax.plot(xvals, yvals, label=label, **style)
            if x_key == "time_s" and np.isfinite(xvals[-1]):
                final_time = float(xvals[-1])
                shared_min_final_time = final_time if shared_min_final_time is None else min(shared_min_final_time, final_time)
            if logy:
                positive_vals.extend(yvals[yvals > 0].tolist())
            else:
                linear_vals.extend(yvals.tolist())

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
            if positive_vals:
                ymin = min(positive_vals)
                ymax = max(positive_vals)
                if ymin > 0:
                    log_min = np.log10(ymin)
                    log_max = np.log10(max(ymax, ymin * 1.01))
                    pad = max(0.1 * (log_max - log_min) if log_max > log_min else 0.25, 0.25)
                    ax.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))
        elif linear_vals:
            ymin = min(linear_vals)
            ymax = max(linear_vals)
            if ymin == ymax:
                delta = abs(ymin) * 0.1 or 1.0
                ax.set_ylim(ymin - delta, ymax + delta)
            else:
                margin = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - margin, ymax + margin)

        x_limits = _configured_x_limits(title, x_key, y_key)
        if isinstance(x_limits, str):
            if x_limits == "shared_min_final" and shared_min_final_time is not None:
                ax.set_xlim(0.0, shared_min_final_time)
        elif x_limits is not None:
            ax.set_xlim(*x_limits)
        y_limits = _configured_y_limits(title, x_key, y_key)
        if y_limits is not None:
            ax.set_ylim(*y_limits)

        if PLOT_SHOW_GRID:
            ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()

        png_name = f"{title.replace(' ', '_').lower()}_{case_name}.png"
        png_path = os.path.join(out_dir, png_name)
        fig.savefig(png_path, dpi=150)
        eps_name = f"{title.replace(' ', '_').lower()}_{case_name}.eps"
        eps_path = os.path.join(out_dir, eps_name)
        fig.savefig(eps_path, format="eps")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        panels.append(Image.open(buf))
        panel_buffers.append(buf)
        plt.close(fig)

    # Keep panel layout aligned by metric: time on the left, iteration on the right.
    render_plot("time_s", "grad_norm", "Gradient norm vs time", "time [s]", "||grad||", logy=True)
    render_plot("iter", "grad_norm", "Gradient norm vs iteration", "iteration", "||grad||", logy=True)
    render_plot("time_s", "f", "Objective vs time", "time [s]", "f", logy=False)
    render_plot("iter", "f", "Objective vs iteration", "iteration", "f", logy=False)
    render_plot("time_s", "objective_gap", "Objective gap vs time", "time [s]", r"$f - f^*$", logy=True)
    render_plot("iter", "objective_gap", "Objective gap vs iteration", "iteration", r"$f - f^*$", logy=True)

    if panels:
        panel_path = os.path.join(out_dir, f"compare_grassmann_panels_{case_name}.png")
        combine_images(panels, panel_path, cols=2)
    for img, buf in zip(panels, panel_buffers):
        img.close()
        buf.close()


def main(argv: List[str]) -> None:
    config = CONFIG
    run_name = config.get("run_name", "grassmann_transport_demo")
    out_dir = os.path.abspath(config.get("output_dir", os.path.join(CODE_DIR, "..", "reports-grassmann", run_name)))
    os.makedirs(out_dir, exist_ok=True)

    problem = config.get("problem", {})
    n = int(problem.get("n", 2000))
    p = int(problem.get("p", 20))
    seed = int(problem.get("seed", 0))
    init_seed = int(problem.get("init_seed", 0))
    a_diag = build_spectrum(problem)
    rng = np.random.default_rng(init_seed)
    Y0, _ = np.linalg.qr(rng.standard_normal((n, p)), mode="reduced")

    methods = config.get("methods", {})
    compare_methods = config.get("compare_methods", list(methods.keys()))
    results: Dict[str, Dict[str, Any]] = {}

    for idx, key in enumerate(compare_methods):
        method_cfg = methods[key]
        opts_dict = dict(method_cfg.get("options", {}))
        opts_dict.setdefault("random_state", seed + idx)
        opts = SolverOptions(**opts_dict)
        csv_path = os.path.join(out_dir, f"{key}.csv")
        if PLOT_ONLY and os.path.exists(csv_path):
            hist = load_history_csv(csv_path)
        else:
            _, hist = solve_grassmann_nystrom_cubic(a_diag, Y0, opts)
            write_history_csv(csv_path, hist)
        results[key] = {
            "hist": hist,
            "label": _format_label(key, method_cfg, opts_dict),
            "csv": csv_path,
        }

    best_final_f = min(results[key]["hist"]["f"][-1] for key in compare_methods)

    histories_for_plot: List[Tuple[str, Dict[str, List[float]]]] = []
    for key in compare_methods:
        hist = results[key]["hist"]
        hist["objective_gap"] = (np.asarray(hist["f"]) - best_final_f).tolist()
        histories_for_plot.append((results[key]["label"], hist))
    plot_histories(run_name, histories_for_plot, out_dir)

    summary_path = os.path.join(out_dir, "compare_grassmann_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "label", "iterations", "time_s", "final_f", "final_grad"])
        writer.writeheader()
        for key in compare_methods:
            hist = results[key]["hist"]
            writer.writerow(
                {
                    "method": key,
                    "label": results[key]["label"],
                    "iterations": len(hist["f"]) - 1,
                    "time_s": hist["time"][-1],
                    "final_f": hist["f"][-1],
                    "final_grad": hist["grad_norm"][-1],
                }
            )


if __name__ == "__main__":
    main(sys.argv[1:])
