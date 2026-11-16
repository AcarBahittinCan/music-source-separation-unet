# train.py
# =============================================================================
# Trains the U-Net:
#
#   forward:  mixture magnitude (B,1,F,T) -> 4 masks (B,4,F,T)
#   estimate: est_mag = masks * mixture_mag        (broadcast over sources)
#   loss:     L1( est_mag , true stem magnitudes )
#
# Run with:   python train.py
# =============================================================================

import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MUSDBDataset
from model import UNet

# ----- Automatic Mixed Precision (AMP) ---------------------------------------
# Unified torch.amp API (PyTorch >= 2.3). Falls back to the legacy
# torch.cuda.amp API on older builds so this file still runs either way.
try:
    from torch.amp import autocast, GradScaler

    def _autocast(enabled):
        return autocast("cuda", enabled=enabled)

    def _make_scaler(enabled):
        return GradScaler("cuda", enabled=enabled)
except (ImportError, TypeError):
    from torch.cuda.amp import autocast, GradScaler          # legacy fallback

    def _autocast(enabled):
        return autocast(enabled=enabled)

    def _make_scaler(enabled):
        return GradScaler(enabled=enabled)

# Hann window for the GPU-side STFT, built once on the target device.
_WINDOW = torch.hann_window(config.WIN_LENGTH, device=config.DEVICE)


def compute_magnitude(waves):
    
    # waves = ( B , cha , samples)
    # lead = (B , cha)
    # flat = (B * cha , samples)
    # spec = (B * cha , F , T)
    # mag  takes phase from spec
        
    lead = waves.shape[:-1]                         # all dims except the time axis 
    flat = waves.reshape(-1, waves.shape[-1])       # (N, samples)
    spec = torch.stft(flat,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    win_length=config.WIN_LENGTH,
                    window=_WINDOW,
                    center=True,
                    return_complex=True)          # (N, F, T)
    mag = spec.abs()
    return mag.reshape(*lead, *mag.shape[-2:])      # (..., F, T) returns the original size with f and t


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    
    
    training = optimizer is not None
    model.train() if training else model.eval()
    use_amp = config.USE_AMP and config.DEVICE.type == "cuda"

    total_loss = 0.0
    seen = 0

    # no_grad during validation saves memory and time
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for mix_wave, stems_wave in tqdm(loader, desc="train" if training else "val  ", leave=False):

            mix_wave   = mix_wave.to(config.DEVICE, non_blocking=True)    # (B, samples)
            stems_wave = stems_wave.to(config.DEVICE, non_blocking=True)  # (B, 4, samples)

            # STFT on the GPU, in fp32, BEFORE entering the autocast region
            mix_mag   = compute_magnitude(mix_wave).unsqueeze(1)   # (B, 1, F, T)
            stems_mag = compute_magnitude(stems_wave)              # (B, 4, F, T)

            if training:
                optimizer.zero_grad()

            # Only the conv-heavy model + loss run in fp16 under autocast
            with _autocast(use_amp):
                masks   = model(mix_mag)        # (B, 4, F, T), softmax over sources
                est_mag = masks * mix_mag       # (B,1,F,T) broadcasts to (B,4,F,T)
                loss    = criterion(est_mag, stems_mag)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()   # scale up to avoid fp16 underflow
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * mix_wave.size(0)
            seen       += mix_wave.size(0)

    return total_loss / seen


def main():
    print(f"Device: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ---- data -----------------------------------------------------------------
    train_set = MUSDBDataset(mode="train")
    val_set   = MUSDBDataset(mode="validation")
    print(f"Train tracks: {len(train_set.tracks)} | Val tracks: {len(val_set.tracks)}")
    print(f"Segments per epoch -> train: {len(train_set)}, val: {len(val_set)}")

    train_loader = DataLoader(train_set,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True,
                            drop_last=True)
    val_loader = DataLoader(val_set,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    # Optional: evaluate the held-out test/ set each epoch (monitoring only).
    test_loader = None
    if config.EVAL_TEST_EACH_EPOCH:
        test_set = MUSDBDataset(mode="test")
        test_loader = DataLoader(test_set,
                                batch_size=config.BATCH_SIZE,
                                shuffle=False,
                                num_workers=config.NUM_WORKERS,
                                pin_memory=True)
        print(f"Test tracks: {len(test_set.tracks)} | "
              f"test segments/epoch: {len(test_set)}")

    # ---- model / loss / optimizer ----------------------------------------------
    model = UNet().to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f} M")

    criterion = nn.L1Loss()                                     # Mean Absolute Error
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Decay the LR when the validation loss plateaus (near a minimum -> take
    # smaller steps). No 'verbose' arg here: it was deprecated in PyTorch 2.x,
    # so we read and print the rate ourselves in the loop instead.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.LR_MIN)

    use_amp = config.USE_AMP and config.DEVICE.type == "cuda"
    scaler  = _make_scaler(use_amp)
    print(f"Mixed precision (AMP): {'ON' if use_amp else 'OFF'}")

    # ---- training loop -----------------------------------------------------------
    best_val = float("inf")
    history = []  # per-epoch metrics, dumped to checkpoints/history.json for evaluate.py
    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer=None)
        test_loss  = (run_epoch(model, test_loader, criterion, optimizer=None)if test_loader is not None else None)

        minutes = (time.time() - t0) / 60

        # LR that was active during THIS epoch, then let the scheduler react to
        # the new val_loss. If it decays, lr_next applies from the NEXT epoch.
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_next = optimizer.param_groups[0]["lr"]
        lr_note = f"  LR decayed -> {lr_next:.2e}" if lr_next < lr_now else ""

        # Save whenever the VALIDATION loss improves -> "best" checkpoint.
        # (Train loss alone can keep dropping while the model overfits.)
        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss},
                       config.BEST_MODEL_PATH)
            tag = "  <- best model saved"

        test_str = f"Test Loss: {test_loss:.5f}, " if test_loss is not None else ""
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}], "
              f"Training Loss: {train_loss:.5f}, "
              f"Validation Loss: {val_loss:.5f}, "
              f"{test_str}"
              f"LR: {lr_now:.1e}  "
              f"({minutes:.1f} min){tag}{lr_note}")

        # Persist per-epoch metrics so evaluate.py can plot the curves later.
        # (Re-written every epoch, so a crash mid-training still leaves a usable file.)
        history.append({"epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "test_loss": test_loss,
                        "lr": lr_now,
                        "minutes": minutes})
        with open(os.path.join(config.CHECKPOINT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best validation loss: {best_val:.5f}")
    print(f"Best weights saved at: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    # This guard is REQUIRED on Windows when NUM_WORKERS > 0:
    # worker processes re-import this file, and without the guard they would
    # each try to start training again.
    main()
