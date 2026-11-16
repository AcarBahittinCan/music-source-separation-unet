# evaluate.py
# =============================================================================
# One-stop evaluation + reporting for the presentation. Run this AFTER training.
# It loads the best checkpoint and produces, in a single output folder:
#
#   PLOTS
#     training_curves.png   - train/val/test loss vs epoch + LR (needs history.json)
#     loss_comparison.png   - final-model loss: overall by split, and by stem
#     sdr_per_stem.png      - median SDR per stem (if an SDR backend is installed)
#     spectrogram_masks.png - mixture spectrogram, the 4 learned masks, the 4 stems
#
#   NUMBERS
#     metrics.csv           - all numeric results in one table
#     history.csv           - per-epoch curve data (converted from history.json)
#     evaluation_report.md  - written summary you can paste into the slides
#
#   AUDIO
#     audio/<track>__mixture.wav + the 4 separated stems  (before/after demo)
#
# It reuses the SAME code paths as training/inference (no duplicated logic):
#   - compute_magnitude  from train.py   (the GPU STFT)
#   - separate_channel   from predict.py (overlap-add inference)
#
# Usage (PowerShell):
#   python evaluate.py
#   python evaluate.py --sdr_tracks 10 --audio_tracks 2
#   python evaluate.py --sdr_tracks 0          # skip the (slow) SDR pass
# =============================================================================

import argparse
import csv
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")              # headless backend: save PNGs without a display
import matplotlib.pyplot as plt

import config
from dataset import MUSDBDataset, list_tracks
from model import UNet
from train import compute_magnitude          # reuse the exact GPU STFT
from predict import separate_channel         # reuse the overlap-add inference


# =============================================================================
# small helpers
# =============================================================================

def _log_mag(x, eps=1e-6):
    """Linear magnitude -> dB scale, which is how spectrograms are normally viewed."""
    return 20.0 * np.log10(np.abs(x) + eps)


def _md_table(headers, rows):
    """Build a GitHub-style markdown table from headers + list-of-rows."""
    line = "| " + " | ".join(headers) + " |"
    sep  = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


# =============================================================================
# 1) FINAL-MODEL LOSS on each split (computed here, no training history needed)
# =============================================================================

@torch.no_grad()
def measure_split_loss(model, dataset, max_segments):
    """
    Run the best model over a fixed subset of a split and return:
      overall L1 (scalar)  and  per-stem L1 (np array of length 4).
    A capped, evenly-spaced subset keeps this fast and makes train/val/test
    comparable (same number of segments each).
    """
    n = min(max_segments, len(dataset))
    idx = np.linspace(0, len(dataset) - 1, n).astype(int).tolist()
    loader = DataLoader(Subset(dataset, idx),
                        batch_size=config.BATCH_SIZE,
                        shuffle=False,
                        num_workers=0)        # tiny subsets -> no workers needed

    model.eval()
    per_stem_sum = torch.zeros(config.NUM_SOURCES, device=config.DEVICE)
    seen = 0
    for mix_wave, stems_wave in loader:
        mix_wave   = mix_wave.to(config.DEVICE)
        stems_wave = stems_wave.to(config.DEVICE)

        mix_mag   = compute_magnitude(mix_wave).unsqueeze(1)   # (B,1,F,T)
        stems_mag = compute_magnitude(stems_wave)              # (B,4,F,T)
        masks     = model(mix_mag)
        est_mag   = masks * mix_mag

        # mean abs error per (sample, source), then sum over the batch
        l1 = (est_mag - stems_mag).abs().mean(dim=(2, 3))      # (B, 4)
        per_stem_sum += l1.sum(dim=0)
        seen += mix_wave.size(0)

    per_stem = (per_stem_sum / seen).cpu().numpy()             # (4,)
    return float(per_stem.mean()), per_stem


# =============================================================================
# 2) SDR  (uses fast_bss_eval if available, else museval; the real quality metric)
# =============================================================================

def _select_sdr_backend():
    """
    Pick an SDR backend. fast_bss_eval is light and needs no ffmpeg; museval is
    the MUSDB-standard (windowed median) but pulls in musdb/stempeg/ffmpeg.
    Catches ANY error (not just ImportError) so a broken install degrades to
    'skip SDR' instead of crashing the whole evaluation.
    Returns (name, module) or (None, None).
    """
    try:
        import fast_bss_eval
        return "fast_bss_eval", fast_bss_eval
    except Exception:
        pass
    try:
        import museval                       # may raise RuntimeError if ffmpeg is missing
        return "museval", museval
    except Exception as e:
        print(f"  No SDR backend usable -> skipping SDR. ({type(e).__name__}: {e})")
        print("  Easiest fix:  pip install fast_bss_eval   (no ffmpeg needed)")
        return None, None


def _track_metrics(backend, mod, refs, est):
    """refs, est: numpy (nsrc, N) mono. Returns (sdr, sir, sar), each (nsrc,) in dB."""
    L = min(refs.shape[1], est.shape[1])
    refs, est = refs[:, :L], est[:, :L]
    if backend == "fast_bss_eval":
        sdr, sir, sar, _perm = mod.bss_eval_sources(refs, est)
        sdr, sir, sar = (np.asarray(x, dtype=float) for x in (sdr, sir, sar))
    else:                                                         # museval: windowed median
        SDR, _ISR, SIR, SAR = mod.evaluate(refs[:, :, None], est[:, :, None])
        sdr = np.nanmedian(SDR, axis=1)
        sir = np.nanmedian(SIR, axis=1)
        sar = np.nanmedian(SAR, axis=1)
    clean = lambda v: np.where(np.isfinite(v), v, np.nan)         # guard inf from silent stems
    return clean(sdr), clean(sir), clean(sar)


def evaluate_sdr(model, n_tracks):
    """
    SDR / SIR / SAR per stem over the first n_tracks of test/ (mono; the model is mono).
      SDR = overall separation quality
      SIR = isolation from the OTHER sources (low SIR => bleed)
      SAR = freedom from artifacts          (low SAR => distortion)
    Returns (metrics, per_track):
      metrics   = {"SDR": {src: median}, "SIR": {...}, "SAR": {...}}
      per_track = [{"track": name, "SDR": vec, "SIR": vec, "SAR": vec}, ...]
    or (None, None) if no backend / no tracks.
    """
    if n_tracks <= 0:
        return None, None
    backend, mod = _select_sdr_backend()
    if backend is None:
        return None, None
    print(f"  SDR backend: {backend}")

    window = torch.hann_window(config.WIN_LENGTH, device=config.DEVICE)
    per_track = []
    for td in list_tracks(config.TEST_DIR)[:n_tracks]:
        try:
            # mixture -> mono -> separate into 4 mono stems
            mix, _ = sf.read(os.path.join(td, "mixture.wav"),
                             dtype="float32", always_2d=True)
            est = separate_channel(model, torch.from_numpy(mix.mean(axis=1)), window).numpy()  # (4,N)
            # ground-truth stems -> mono, same order as config.SOURCES
            refs = np.stack([sf.read(os.path.join(td, f"{name}.wav"),
                                     dtype="float32", always_2d=True)[0].mean(axis=1)
                             for name in config.SOURCES], axis=0)                               # (4,N)
            sdr, sir, sar = _track_metrics(backend, mod, refs, est)
            per_track.append({"track": os.path.basename(td),
                              "SDR": sdr, "SIR": sir, "SAR": sar})
            print("  " + os.path.basename(td)[:34].ljust(34) + "  "
                  + "  ".join(f"{n}:{v:5.2f}" for n, v in zip(config.SOURCES, sdr)) + "  (SDR dB)")
        except Exception as e:
            print(f"  (skipped {os.path.basename(td)}: {e})")

    if not per_track:
        return None, None
    metrics = {}
    for key in ("SDR", "SIR", "SAR"):
        stacked = np.stack([t[key] for t in per_track], axis=0)        # (ntracks, nsrc)
        med = np.nanmedian(stacked, axis=0)                            # (nsrc,)
        metrics[key] = {name: float(v) for name, v in zip(config.SOURCES, med)}
    return metrics, per_track


# =============================================================================
# 3) Example masks for the spectrogram figure
# =============================================================================

@torch.no_grad()
def example_masks(model):
    """Take one deterministic test segment and return (mix_mag, masks, stem_mags) as numpy."""
    ds = MUSDBDataset(mode="test")
    mix_wave, _stems = ds[0]                                  # (samples,), (4, samples)
    mix_wave = mix_wave.unsqueeze(0).to(config.DEVICE)        # (1, samples)

    mix_mag = compute_magnitude(mix_wave).unsqueeze(1)        # (1,1,F,T)
    masks   = model(mix_mag)                                  # (1,4,F,T)
    est_mag = masks * mix_mag                                 # (1,4,F,T)
    return (mix_mag[0, 0].cpu().numpy(),                      # (F,T)
            masks[0].cpu().numpy(),                           # (4,F,T)
            est_mag[0].cpu().numpy())                         # (4,F,T)


# =============================================================================
# 4) AUDIO export (before/after demo)
# =============================================================================

def export_audio(model, n_tracks, out_dir):
    """Fully separate the first n_tracks of test/ and write mixture + 4 stems (stereo)."""
    if n_tracks <= 0:
        return []
    window = torch.hann_window(config.WIN_LENGTH, device=config.DEVICE)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    written = []
    for td in list_tracks(config.TEST_DIR)[:n_tracks]:
        name = os.path.basename(td)
        mix, _ = sf.read(os.path.join(td, "mixture.wav"),
                         dtype="float32", always_2d=True)     # (samples, ch)
        # process each channel independently -> keep stereo (same as predict.py)
        per_ch = [separate_channel(model, torch.from_numpy(mix[:, c]), window)
                  for c in range(mix.shape[1])]
        stems = torch.stack(per_ch, dim=-1).numpy()           # (4, samples, ch)

        safe = name.replace(" ", "_").replace("/", "_")
        sf.write(os.path.join(audio_dir, f"{safe}__mixture.wav"), mix, config.SAMPLE_RATE)
        for i, s in enumerate(config.SOURCES):
            sf.write(os.path.join(audio_dir, f"{safe}__{s}.wav"),
                     stems[i], config.SAMPLE_RATE)
        written.append(name)
        print(f"  exported audio for: {name}")
    return written


# =============================================================================
# PLOTTING (each takes plain numpy/dicts so it's easy to reason about)
# =============================================================================

def plot_training_curves(history, out_path):
    """train/val/test loss vs epoch (top) and learning rate vs epoch (bottom)."""
    epochs = [h["epoch"] for h in history]
    train  = [h["train_loss"] for h in history]
    val    = [h["val_loss"] for h in history]
    test   = [h.get("test_loss") for h in history]
    lr     = [h["lr"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(epochs, train, marker="o", ms=3, label="Train")
    ax1.plot(epochs, val,   marker="s", ms=3, label="Validation")
    if any(t is not None for t in test):
        te = [(e, t) for e, t in zip(epochs, test) if t is not None]
        ax1.plot([e for e, _ in te], [t for _, t in te],
                 marker="^", ms=3, label="Test")

    best_i = int(np.argmin(val))                       # best validation epoch
    ax1.axvline(epochs[best_i], color="gray", ls="--", lw=1)
    ax1.annotate(f"best val (epoch {epochs[best_i]})",
                 xy=(epochs[best_i], val[best_i]),
                 xytext=(6, 8), textcoords="offset points", fontsize=9)
    ax1.set_ylabel("L1 loss (magnitude)")
    ax1.set_title("Training / Validation / Test Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, lr, color="tab:red", marker=".")
    ax2.set_yscale("log")
    ax2.set_ylabel("Learning rate")
    ax2.set_xlabel("Epoch")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_loss_comparison(split_overall, split_per_stem, sources, out_path):
    """Left: overall loss per split. Right: per-stem loss grouped by split."""
    splits = list(split_overall.keys())
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # overall, one bar per split (the train-vs-test generalization gap at a glance)
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    axA.bar(splits, [split_overall[s] for s in splits],
            color=[colors.get(s, "tab:gray") for s in splits])
    for i, s in enumerate(splits):
        axA.text(i, split_overall[s], f"{split_overall[s]:.4f}",
                 ha="center", va="bottom", fontsize=9)
    axA.set_ylabel("L1 loss (magnitude)")
    axA.set_title("Final-model overall loss by split")
    axA.grid(axis="y", alpha=0.3)

    # per-stem, grouped bars (which stems are hard, and the gap per stem)
    x = np.arange(len(sources))
    w = 0.8 / len(splits)
    for i, s in enumerate(splits):
        axB.bar(x + i * w, split_per_stem[s], width=w,
                label=s, color=colors.get(s, "tab:gray"))
    axB.set_xticks(x + w * (len(splits) - 1) / 2)
    axB.set_xticklabels(sources)
    axB.set_ylabel("L1 loss (magnitude)")
    axB.set_title("Final-model loss by stem")
    axB.legend()
    axB.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sdr(sdr, out_path):
    """Median SDR per stem (higher = better separation)."""
    sources = list(sdr.keys())
    vals = [sdr[s] for s in sources]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(sources, vals, color="tab:purple")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Median SDR (dB)")
    ax.set_title("Separation quality per stem (SDR, higher is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_spectrograms(mix_mag, masks, stem_mags, sources, out_path):
    """
    Mixture spectrogram (top, spanning), the 4 learned masks (middle row),
    and the 4 separated stem spectrograms (bottom row). This literally
    pictures WHAT the model learned - the masks are its decision.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, len(sources))

    ax_mix = fig.add_subplot(gs[0, :])
    ax_mix.imshow(_log_mag(mix_mag), origin="lower", aspect="auto", cmap="magma")
    ax_mix.set_title("Mixture magnitude spectrogram (dB)")
    ax_mix.set_ylabel("Freq bin")

    for i, name in enumerate(sources):
        axm = fig.add_subplot(gs[1, i])
        im = axm.imshow(masks[i], origin="lower", aspect="auto",
                        cmap="viridis", vmin=0, vmax=1)
        axm.set_title(f"mask: {name}")
        if i == 0:
            axm.set_ylabel("Freq bin")

        axs = fig.add_subplot(gs[2, i])
        axs.imshow(_log_mag(stem_mags[i]), origin="lower", aspect="auto", cmap="magma")
        axs.set_title(f"separated: {name}")
        axs.set_xlabel("Time frame")
        if i == 0:
            axs.set_ylabel("Freq bin")

    fig.colorbar(im, ax=fig.get_axes(), fraction=0.015, pad=0.01,
                 label="mask value (0-1)")
    fig.suptitle("What the model learned: mixture -> masks -> separated stems",
                 fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# REPORT + CSV writers
# =============================================================================

def write_history_csv(history, out_path):
    keys = ["epoch", "train_loss", "val_loss", "test_loss", "lr", "minutes"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for h in history:
            w.writerow([h.get(k) for k in keys])


def write_metrics_csv(split_overall, split_per_stem, metrics, sources, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + sources + ["overall"])
        for split in split_overall:
            w.writerow([f"{split}_L1"]
                       + [f"{v:.6f}" for v in split_per_stem[split]]
                       + [f"{split_overall[split]:.6f}"])
        if metrics is not None:
            for key in ("SDR", "SIR", "SAR"):
                w.writerow([f"{key}_dB"] + [f"{metrics[key][s]:.3f}" for s in sources] + [""])


def write_per_track_csv(per_track, sources, out_path):
    """Every track's SDR/SIR/SAR - useful for showing the spread (a 5-track median hides it)."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "metric"] + sources)
        for t in per_track:
            for key in ("SDR", "SIR", "SAR"):
                w.writerow([t["track"], key] + [f"{v:.3f}" for v in t[key]])


def write_report(out_dir, n_params, split_overall, split_per_stem,
                 metrics, sources, ckpt_info, history, generated):
    lines = []
    lines.append("# Music Source Separation - Baseline U-Net: Evaluation Report\n")

    # --- model + config snapshot ---
    lines.append("## Model & configuration\n")
    lines.append(f"- Parameters: **{n_params:.2f} M**")
    lines.append(f"- Architecture: spectrogram U-Net, 4 softmax masks (stems sum to mixture)")
    lines.append(f"- Sample rate / n_fft / hop: {config.SAMPLE_RATE} / {config.N_FFT} / {config.HOP_LENGTH}")
    lines.append(f"- Segment length: {config.SEGMENT_SECONDS}s | Batch size: {config.BATCH_SIZE} | "
                 f"Base channels: {config.BASE_CHANNELS}")
    lines.append(f"- Optimizer: Adam (lr {config.LEARNING_RATE}), L1 magnitude loss, "
                 f"AMP={'on' if config.USE_AMP else 'off'}\n")

    # --- training summary ---
    lines.append("## Training summary\n")
    if ckpt_info:
        lines.append(f"- Best checkpoint: epoch **{ckpt_info['epoch']}**, "
                     f"validation loss **{ckpt_info['val_loss']:.5f}**")
    if history:
        total_min = sum(h.get("minutes", 0) for h in history)
        lines.append(f"- Epochs run: {len(history)} | Total training time: ~{total_min:.1f} min "
                     f"(~{total_min/max(len(history),1):.1f} min/epoch)")
    lines.append("")

    # --- loss tables ---
    lines.append("## Final-model loss (L1 on magnitude spectrogram, lower = better)\n")
    rows = []
    for split in split_overall:
        rows.append([split]
                    + [f"{v:.4f}" for v in split_per_stem[split]]
                    + [f"{split_overall[split]:.4f}"])
    lines.append(_md_table(["split"] + sources + ["overall"], rows))
    lines.append("")

    # --- SDR / SIR / SAR table ---
    lines.append("## Separation quality - SDR / SIR / SAR (dB, higher = better)\n")
    if metrics is not None:
        rows = [[s, f"{metrics['SDR'][s]:.2f}", f"{metrics['SIR'][s]:.2f}", f"{metrics['SAR'][s]:.2f}"]
                for s in sources]
        lines.append(_md_table(["stem", "SDR", "SIR", "SAR"], rows))
        lines.append("")
        lines.append("_SDR = overall quality; SIR = isolation from other sources "
                     "(low => bleed); SAR = freedom from artifacts (low => distortion)._")
    else:
        lines.append("_SDR not computed (no backend available or skipped). "
                     "Re-run with `--sdr_tracks N` after `pip install fast_bss_eval`._")
    lines.append("")

    # --- notes for the slides ---
    lines.append("## Notes for interpretation\n")
    lines.append("- **Train vs val/test gap** measures generalization, not memorization: "
                 "random cropping prevents segment-level memorization, but only 92 training "
                 "songs caps how well it transfers to unseen music.")
    lines.append("- **L1 magnitude loss is not a perceptual metric.** It shows the model trained "
                 "and generalized; **SDR** (on reconstructed audio) is the real quality number and "
                 "the right basis for the baseline-vs-SpecBandNet comparison.")
    lines.append("- Expect **vocals/drums** to score better than **bass/other** - a known, "
                 "explainable property of a small from-scratch model.\n")

    # --- file index ---
    lines.append("## Generated files\n")
    for g in generated:
        lines.append(f"- `{g}`")
    lines.append("")

    path = os.path.join(out_dir, "evaluation_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate + report on a trained model")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--output_dir", default="evaluation_output")
    parser.add_argument("--history", default=os.path.join(config.CHECKPOINT_DIR, "history.json"))
    parser.add_argument("--loss_segments", type=int, default=64,
                        help="segments per split for the loss snapshot")
    parser.add_argument("--sdr_tracks", type=int, default=5,
                        help="test tracks to score with museval (0 = skip; slow)")
    parser.add_argument("--audio_tracks", type=int, default=1,
                        help="test tracks to export as before/after audio (0 = skip)")
    args = parser.parse_args()

    # reproducibility (the train split uses random crops; seeding pins the snapshot)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {config.DEVICE}")

    # ---- load the trained model ------------------------------------------------
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"No checkpoint at '{args.checkpoint}'. Train first.")
    model = UNet().to(config.DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.5f} "
          f"({n_params:.2f} M params)\n")

    generated = []

    # ---- 1) final-model loss per split + per stem ------------------------------
    print("Measuring final-model loss on each split...")
    split_overall, split_per_stem = {}, {}
    for split in ("train", "validation", "test"):
        try:
            ds = MUSDBDataset(mode=split)
            ov, ps = measure_split_loss(model, ds, args.loss_segments)
            split_overall[split] = ov
            split_per_stem[split] = ps
            print(f"  {split:<11} overall L1 = {ov:.5f}  |  "
                  + ", ".join(f"{n}:{v:.4f}" for n, v in zip(config.SOURCES, ps)))
        except Exception as e:
            print(f"  ({split} skipped: {e})")

    # ---- 2) SDR / SIR / SAR ----------------------------------------------------
    print("\nComputing SDR/SIR/SAR (this is the slow part)...")
    metrics, per_track = evaluate_sdr(model, args.sdr_tracks)

    # ---- plots -----------------------------------------------------------------
    print("\nWriting plots...")
    if split_overall:
        p = os.path.join(args.output_dir, "loss_comparison.png")
        plot_loss_comparison(split_overall, split_per_stem, config.SOURCES, p)
        generated.append("loss_comparison.png")
    if metrics is not None:
        p = os.path.join(args.output_dir, "sdr_per_stem.png")
        plot_sdr(metrics["SDR"], p)
        generated.append("sdr_per_stem.png")
        write_per_track_csv(per_track, config.SOURCES,
                            os.path.join(args.output_dir, "sdr_per_track.csv"))
        generated.append("sdr_per_track.csv")

    # spectrogram / mask figure
    try:
        mix_mag, masks, stem_mags = example_masks(model)
        p = os.path.join(args.output_dir, "spectrogram_masks.png")
        plot_spectrograms(mix_mag, masks, stem_mags, config.SOURCES, p)
        generated.append("spectrogram_masks.png")
    except Exception as e:
        print(f"  (spectrogram figure skipped: {e})")

    # training curves (need history.json from train.py)
    history = []
    if os.path.exists(args.history):
        with open(args.history) as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))
        write_history_csv(history, os.path.join(args.output_dir, "history.csv"))
        generated += ["training_curves.png", "history.csv"]
    else:
        print(f"  (no {args.history} -> skipping epoch curves. "
              f"Re-run train.py to generate it.)")

    # ---- 3) audio export -------------------------------------------------------
    if args.audio_tracks > 0:
        print("\nExporting before/after audio...")
        names = export_audio(model, args.audio_tracks, args.output_dir)
        if names:
            generated.append("audio/  (mixture + 4 stems per track)")

    # ---- CSV + report ----------------------------------------------------------
    if split_overall:
        write_metrics_csv(split_overall, split_per_stem, metrics, config.SOURCES,
                          os.path.join(args.output_dir, "metrics.csv"))
        generated.append("metrics.csv")

    report_path = write_report(args.output_dir, n_params, split_overall, split_per_stem,
                               metrics, config.SOURCES, ckpt, history, generated)
    generated.append(os.path.basename(report_path))

    print(f"\nDone. Everything written to:  {os.path.abspath(args.output_dir)}")
    for g in generated:
        print(f"   - {g}")


if __name__ == "__main__":
    main()