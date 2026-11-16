

import os
import torch

DATASET_PATH = os.environ.get("MUSDB_PATH", "musdb18hq")   # set MUSDB_PATH or place the dataset here

TRAIN_DIR = os.path.join(DATASET_PATH, "train")   
TEST_DIR  = os.path.join(DATASET_PATH, "test")    

CHECKPOINT_DIR  = "checkpoints"                                   
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_unet_baseline.pth")
OUTPUT_DIR      = "separated_output"                           


# -----------------------------------------------------------------------------
# AUDIO / STFT PARAMETERS
# -----------------------------------------------------------------------------
SAMPLE_RATE = 44100   # MUSDB18-HQ native rate (Hz)
N_FFT       = 2048    # FFT size -> N_FFT/2 + 1 = 1025 frequency bins
HOP_LENGTH  = 512     # 75% overlap between consecutive STFT frames
WIN_LENGTH  = 2048    # analysis window length (= N_FFT here, Hann window)

# -----------------------------------------------------------------------------
# DATA / SEGMENTATION
# -----------------------------------------------------------------------------
SEGMENT_SECONDS = 5.0                                   # length of each training chunk
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SAMPLE_RATE)    # 132300 samples
                                                        # -> spectrogram of ~259 frames

SOURCES     = ["vocals", "drums", "bass", "other"]      # output order of the 4 masks
NUM_SOURCES = len(SOURCES)

VAL_TRACKS              = 8    # last 8 (alphabetical) train tracks become validation
SEGMENTS_PER_TRACK      = 32   # random crops drawn from each TRAIN track per epoch
VAL_SEGMENTS_PER_TRACK  = 8   # fixed, evenly spaced crops per VALIDATION track
TEST_SEGMENTS_PER_TRACK = 4    # fixed crops per TEST track. test/ has 50 tracks, so
                               # keep this small or per-epoch test eval gets slow.

EVAL_TEST_EACH_EPOCH = True    # also report loss on the held-out test/ set every
                               # epoch (monitoring only). Set False to skip it and
                               # speed up training / quick smoke tests.

# -----------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
BATCH_SIZE    = 6      # with AMP on there is headroom to try 6-8; drop to 2 on OOM
LEARNING_RATE = 1e-3   # Adam default for spectrogram U-Nets
NUM_EPOCHS    = 50
NUM_WORKERS   = 2      # DataLoader workers; set to 0 if it hangs on Windows
BASE_CHANNELS = 48     # width of first U-Net level (doubles at every depth)

# Learning-rate decay (ReduceLROnPlateau): when validation loss stops improving
# the optimizer is near a minimum, so we shrink the step size to settle into it
# rather than bounce around. LR is multiplied by LR_FACTOR after LR_PATIENCE
# epochs with no improvement, and never drops below LR_MIN.
LR_FACTOR   = 0.5      # halve the learning rate on each plateau
LR_PATIENCE = 3        # epochs of no val-loss improvement before decaying
LR_MIN      = 1e-6     # floor; stop decaying once we reach this

USE_AMP = True         # Automatic Mixed Precision: fp16 forward/backward on
                       # Tensor Cores -> ~half the VRAM, 1.5-2x faster.
                       # Set to False when debugging numerical issues.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




PREDICT_TRACK      = "tame-impala-let-it-happen"   # which test/ track to separate
PREDICT_INPUT      = ""                                  # or a full .wav path (overrides the above)
PREDICT_OUTPUT_DIR = OUTPUT_DIR                           # parent folder for results