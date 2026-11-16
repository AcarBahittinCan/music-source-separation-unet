# predict.py
# =============================================================================
# Loads the trained U-Net and separates a full song into 4 stem wav files.
#
# THE RECONSTRUCTION TRICK:
# The model only predicts MAGNITUDE masks - it knows nothing about phase.
# To get a waveform back, we combine the estimated magnitude with the PHASE
# of the original MIXTURE (a standard, surprisingly effective approximation):
#
#     est_complex = est_magnitude * exp(j * mixture_phase)
#     waveform    = iSTFT(est_complex)
#
# STEREO: the model is mono, so we run it on the left and right channels
# independently and stack the two results back into a stereo file.
#
# Usage (PowerShell):
#   python predict.py                                  # uses config.PREDICT_TRACK / PREDICT_INPUT
#   python predict.py --input "path\to\mixture.wav"    # or override on the CLI
# =============================================================================

import argparse
import os

import soundfile as sf
import torch

import config
from model import UNet

# Sliding-window inference: run the model on overlapping CHUNK_FRAMES-wide
# windows (~12 s each) and average the mask predictions where windows overlap.
# Overlap removes the mask discontinuity that non-overlapping chunks leave at
# their borders, for ~2x the inference compute. HOP = CHUNK/2 gives 50% overlap.
CHUNK_FRAMES = 1024
HOP_FRAMES   = 512


def separate_channel(model, wave, window):
    """
    wave: 1-D float32 tensor (samples,) holding ONE audio channel.
    Returns: (4, samples) tensor - the four separated stems for that channel.
    """
    wave = wave.to(config.DEVICE)

    # ---- 1) STFT of the mixture channel ------------------------------------
    spec = torch.stft(wave,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    win_length=config.WIN_LENGTH,
                    window=window,
                    center=True,
                    return_complex=True)        # (F, T) complex
    mag   = spec.abs()                            # what the model sees
    phase = torch.angle(spec)                     # kept for reconstruction

    # ---- 2) run the model with 50% overlap, averaging the predictions ------
    n_freq, n_time = mag.shape
    acc   = torch.zeros((config.NUM_SOURCES, n_freq, n_time), device=config.DEVICE)
    count = torch.zeros((1, 1, n_time), device=config.DEVICE)   # windows hitting each frame

    for start in range(0, n_time, HOP_FRAMES):
        end   = min(start + CHUNK_FRAMES, n_time)
        chunk = mag[:, start:end]                              # (F, t)
        with torch.no_grad():
            # add batch + channel dims -> (1, 1, F, t); model pads internally
            m = model(chunk.unsqueeze(0).unsqueeze(0)).squeeze(0)   # (4, F, t)
        acc[:, :, start:end]   += m
        count[:, :, start:end] += 1
        if end == n_time:          # window reached the track end -> done
            break

    # Each window's masks sum to 1 across sources, so the average does too.
    masks = acc / count                                        # (4, F, T)

    # ---- 3) apply masks, attach the mixture phase ---------------------------
    est_mag = masks * mag.unsqueeze(0)                    # (4, F, T)
    # torch.polar(abs, angle) builds a complex tensor: abs * exp(j * angle)
    est_complex = torch.polar(est_mag,
                              phase.unsqueeze(0).expand_as(est_mag))

    # ---- 4) inverse STFT back to time-domain waveforms ----------------------
    stems = torch.istft(est_complex,                      # batched over 4 sources
                        n_fft=config.N_FFT,
                        hop_length=config.HOP_LENGTH,
                        win_length=config.WIN_LENGTH,
                        window=window,
                        center=True,
                        length=wave.shape[0])             # (4, samples)
    return stems.cpu()


def resolve_io(input_arg, output_arg):
    """
    Decide WHICH mixture to separate and WHERE to write the stems, using config.py
    as the default so you can avoid command-line flags entirely.
    Input priority:  --input flag  >  config.PREDICT_INPUT  >  config.PREDICT_TRACK
    The output goes in a per-track SUBFOLDER so one song never overwrites another.
    """
    if input_arg:
        input_path = input_arg                                   # CLI flag wins
    elif config.PREDICT_INPUT:
        input_path = config.PREDICT_INPUT                        # explicit path in config
    else:
        input_path = os.path.join(config.TEST_DIR,               # just a track name in config
                                  config.PREDICT_TRACK, "mixture.wav")

    # Name the output folder after the track: ".../<Track>/mixture.wav" -> "<Track>",
    # otherwise fall back to the wav's own filename.
    parent = os.path.basename(os.path.dirname(input_path))
    stem   = os.path.splitext(os.path.basename(input_path))[0]
    label  = parent if (stem == "mixture" and parent) else stem
    out_dir = os.path.join(output_arg, label.replace(" ", "_"))
    return input_path, out_dir


def main():
    parser = argparse.ArgumentParser(description="Separate a song into 4 stems")
    parser.add_argument("--input", default="",
                        help="mixture .wav to separate (overrides config.PREDICT_TRACK)")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH,
                        help="path to the trained .pth file")
    parser.add_argument("--output_dir", default=config.PREDICT_OUTPUT_DIR,
                        help="parent folder; each track gets its own subfolder here")
    args = parser.parse_args()

    # Resolve input/output from config (or the CLI overrides above).
    input_path, out_dir = resolve_io(args.input, args.output_dir)

    # ---- load model ----------------------------------------------------------
    model = UNet().to(config.DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=config.DEVICE,
                    weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()                                  # freeze BatchNorm statistics
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
        f"(val loss {ckpt['val_loss']:.5f})")

    # ---- load audio, keep channels separate ----------------------------------
    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)  # (samples, ch)
    if sr != config.SAMPLE_RATE:
        raise ValueError(f"Expected {config.SAMPLE_RATE} Hz, got {sr} Hz. "
                        f"Resample the file first.")
    print(f"Input: {input_path}  "
        f"({audio.shape[0] / sr:.1f} s, {audio.shape[1]} channel(s))")

    window = torch.hann_window(config.WIN_LENGTH, device=config.DEVICE)

    # ---- separate each channel independently, then rebuild stereo ------------
    per_channel = []
    for ch in range(audio.shape[1]):
        wave = torch.from_numpy(audio[:, ch])
        per_channel.append(separate_channel(model, wave, window))  # (4, samples)
    stems = torch.stack(per_channel, dim=-1).numpy()               # (4, samples, ch)

    # ---- write one wav per source ---------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(config.SOURCES):
        out_path = os.path.join(out_dir, f"{name}.wav")
        sf.write(out_path, stems[i], config.SAMPLE_RATE)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()