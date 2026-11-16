# compare.py
# =============================================================================
# Put several runs side by side. Each "run" is a folder that contains a
# metrics.csv (either directly, or inside an evaluation_output/ subfolder) -
# exactly what evaluate.py writes. Reads them, prints a comparison table, and
# saves a grouped bar chart + a comparison.csv.
#
# This DOES NOT touch the training/inference pipeline - it only reads the
# metrics files those runs already produced. Works for comparing config
# variations now, and for baseline-vs-SpecBandNet later (same metrics.csv format).
#
# Usage (PowerShell):
#   python compare.py runs\baseline_b4_c32_s3 runs\tuned_b6_c48_s5
#   python compare.py                       # auto-discovers every runs\*\ folder
#   python compare.py --metric SIR_dB       # chart SIR instead of SDR
# =============================================================================

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir):
    """
    Read a run's metrics.csv -> (label, data, sources).
    data maps each row name ("SDR_dB", "test_L1", ...) to {column: float}.
    Looks for metrics.csv directly in run_dir or in run_dir/evaluation_output/.
    """
    candidates = [os.path.join(run_dir, "metrics.csv"),
                  os.path.join(run_dir, "evaluation_output", "metrics.csv")]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        print(f"  (no metrics.csv under {run_dir} -> skipped)")
        return None

    with open(path) as f:
        rows = list(csv.reader(f))
    header  = rows[0]                       # ['metric', 'vocals', ..., 'overall']
    sources = header[1:-1]                  # drop 'metric' and 'overall'
    data = {}
    for r in rows[1:]:
        if not r:
            continue
        vals = {}
        for col, cell in zip(header[1:], r[1:]):
            try:
                vals[col] = float(cell)
            except (ValueError, TypeError):
                vals[col] = np.nan
        data[r[0]] = vals
    label = os.path.basename(os.path.normpath(run_dir))
    return label, data, sources


def main():
    ap = argparse.ArgumentParser(description="Compare runs' metrics side by side")
    ap.add_argument("runs", nargs="*", help="run folders (default: every runs\\*\\)")
    ap.add_argument("--metric", default="SDR_dB",
                    help="which metric row to chart (SDR_dB, SIR_dB, SAR_dB, test_L1)")
    ap.add_argument("--output_dir", default="comparison")
    args = ap.parse_args()

    run_dirs = args.runs or sorted(glob.glob(os.path.join("runs", "*", "")))
    if not run_dirs:
        print("No runs found. Pass run folders, or save some into runs\\<name>\\ first.")
        return

    loaded = [m for m in (load_metrics(r) for r in run_dirs) if m]
    if not loaded:
        print("Nothing to compare.")
        return
    sources = loaded[0][2]
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- table to console + comparison.csv ---------------------------------
    out_csv = os.path.join(args.output_dir, "comparison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run"] + sources + ["test_L1"])
        for label, data, _ in loaded:
            metric = data.get(args.metric, {})
            l1     = data.get("test_L1", {})
            w.writerow([label]
                       + [f"{metric.get(s, float('nan')):.3f}" for s in sources]
                       + [f"{l1.get('overall', float('nan')):.4f}"])

    print(f"\n{args.metric} per stem (and overall test L1), higher SDR = better:\n")
    print("  " + "run".ljust(26) + "".join(s[:7].rjust(9) for s in sources) + "   test_L1")
    for label, data, _ in loaded:
        metric = data.get(args.metric, {})
        l1     = data.get("test_L1", {})
        print("  " + label[:26].ljust(26)
              + "".join(f"{metric.get(s, float('nan')):9.2f}" for s in sources)
              + f"   {l1.get('overall', float('nan')):.4f}")

    # ---- grouped bar chart: metric per stem, one group of bars per run -----
    x = np.arange(len(sources))
    bw = 0.8 / len(loaded)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(sources)), 5))
    for i, (label, data, _) in enumerate(loaded):
        metric = data.get(args.metric, {})
        ax.bar(x + i * bw, [metric.get(s, np.nan) for s in sources],
               width=bw, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + bw * (len(loaded) - 1) / 2)
    ax.set_xticklabels(sources)
    ax.set_ylabel(f"{args.metric} (dB)")
    ax.set_title(f"{args.metric} by stem, across runs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, f"comparison_{args.metric.lower()}.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    print(f"\nSaved:  {out_csv}\n        {out_png}")


if __name__ == "__main__":
    main()