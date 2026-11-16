# dataset.py
# =============================================================================
# Loads MUSDB18-HQ from disk and serves raw (mixture, stems) WAVEFORM pairs.
#
# Pipeline for ONE training sample:
#   1. pick a track and a start offset
#   2. read ONLY the needed 3-second slice of each wav (fast partial read,
#      so the 22 GB dataset never has to fit in RAM)
#   3. downmix stereo -> mono
#
# NOTE: the STFT is intentionally NOT done here. Computing spectrograms inside
# DataLoader workers (on the CPU) is the classic source-separation bottleneck:
# the GPU sits idle waiting on complex math. Instead this Dataset stays trivial
# (just a disk read) and train.py runs ONE batched torch.stft on the GPU per
# step - same result, no extra disk, and random cropping is preserved as
# augmentation. Phase is reused from the mixture at inference time (predict.py).
# =============================================================================

import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

import config


def list_tracks(root_dir):
    """Return a sorted list of track folder paths inside root_dir."""
    tracks = [os.path.join(root_dir, d)
              for d in sorted(os.listdir(root_dir))
              if os.path.isdir(os.path.join(root_dir, d))]
    if len(tracks) == 0:
        raise FileNotFoundError(
            f"No track folders found in '{root_dir}'. Check config.DATASET_PATH.")
    return tracks


def load_segment(wav_path, start_sample, num_samples):
    """
    Read the slice [start, start + num_samples) of a wav file and downmix
    to mono. Returns a 1-D float32 torch tensor of length num_samples.
    """
    # soundfile reads ONLY the requested frames from disk -> very fast
    audio, _ = sf.read(wav_path,
                       start=start_sample,
                       stop=start_sample + num_samples,
                       dtype="float32",
                       always_2d=True)        # shape (samples, channels)
    audio = audio.mean(axis=1)                # stereo -> mono

    # Zero-pad if the slice came up short (can happen right at a track's end)
    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)))
    return torch.from_numpy(audio)


class MUSDBDataset(Dataset):
    """
    mode="train"      -> a NEW random offset every time a segment is drawn.
                         This acts as data augmentation (random cropping), so
                         each epoch effectively sees different audio chunks.
    mode="validation" -> fixed, evenly spaced offsets, so the validation loss
                         is computed on the SAME audio every epoch and stays
                         comparable across epochs.
    mode="test"       -> same fixed-offset scheme as validation, but drawn from
                         the held-out test/ folder. For monitoring only.
    """

    def __init__(self, mode="train"):
        assert mode in ("train", "validation", "test")
        self.mode = mode

        if mode == "test":
            # The held-out test/ folder (all tracks), evaluated for monitoring.
            self.tracks = list_tracks(config.TEST_DIR)
            self.segments_per_track = config.TEST_SEGMENTS_PER_TRACK
        else:
            all_tracks = list_tracks(config.TRAIN_DIR)
            # Deterministic split: last VAL_TRACKS alphabetical tracks = validation
            if mode == "train":
                self.tracks = all_tracks[:-config.VAL_TRACKS]
                self.segments_per_track = config.SEGMENTS_PER_TRACK
            else:  # validation
                self.tracks = all_tracks[-config.VAL_TRACKS:]
                self.segments_per_track = config.VAL_SEGMENTS_PER_TRACK

        # Cache each track's length (in samples) ONCE, so __getitem__ can pick
        # valid offsets without re-opening file headers on every call.
        self.track_lengths = []
        for t in self.tracks:
            info = sf.info(os.path.join(t, "mixture.wav"))
            self.track_lengths.append(info.frames)

    def __len__(self):
        # Virtual epoch size: every track contributes a fixed number of segments
        return len(self.tracks) * self.segments_per_track

    def __getitem__(self, idx):
        # Map the flat index -> (which track, which segment of that track)
        track_idx = idx // self.segments_per_track
        seg_idx   = idx %  self.segments_per_track

        track     = self.tracks[track_idx]
        max_start = max(0, self.track_lengths[track_idx] - config.SEGMENT_SAMPLES)

        if self.mode == "train":
            # Random crop -> different audio every epoch
            start = random.randint(0, max_start)
        else:
            # Evenly spaced deterministic crops across the track
            if self.segments_per_track > 1:
                step = max_start // (self.segments_per_track - 1)
            else:
                step = 0
            start = seg_idx * step

        # ---- mixture: model input (raw waveform) ----------------------------
        mix_wave = load_segment(os.path.join(track, "mixture.wav"),
                                start, config.SEGMENT_SAMPLES)     # (samples,)

        # ---- the 4 target stems: training targets ---------------------------
        stem_waves = []
        for name in config.SOURCES:
            stem_waves.append(load_segment(os.path.join(track, f"{name}.wav"),
                                           start, config.SEGMENT_SAMPLES))
        stems_wave = torch.stack(stem_waves, dim=0)               # (4, samples)

        return mix_wave, stems_wave
