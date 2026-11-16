# Music Source Separation - Baseline U-Net: Evaluation Report

## Model & configuration

- Parameters: **7.76 M**
- Architecture: spectrogram U-Net, 4 softmax masks (stems sum to mixture)
- Sample rate / n_fft / hop: 44100 / 2048 / 512
- Segment length: 3.0s | Batch size: 4 | Base channels: 32
- Optimizer: Adam (lr 0.001), L1 magnitude loss, AMP=on

## Training summary

- Best checkpoint: epoch **43**, validation loss **0.11551**
- Epochs run: 50 | Total training time: ~595.4 min (~11.9 min/epoch)

## Final-model loss (L1 on magnitude spectrogram, lower = better)

| split | vocals | drums | bass | other | overall |
| --- | --- | --- | --- | --- | --- |
| train | 0.1017 | 0.1114 | 0.0590 | 0.1290 | 0.1003 |
| validation | 0.1161 | 0.1029 | 0.0784 | 0.1646 | 0.1155 |
| test | 0.0599 | 0.0827 | 0.0345 | 0.0960 | 0.0683 |

## Separation quality - SDR / SIR / SAR (dB, higher = better)

| stem | SDR | SIR | SAR |
| --- | --- | --- | --- |
| vocals | 5.67 | 13.77 | 6.60 |
| drums | 4.95 | 11.16 | 6.64 |
| bass | 2.32 | 5.16 | 6.50 |
| other | 2.20 | 6.41 | 5.44 |

_SDR = overall quality; SIR = isolation from other sources (low => bleed); SAR = freedom from artifacts (low => distortion)._

## Notes for interpretation

- **Train vs val/test gap** measures generalization, not memorization: random cropping prevents segment-level memorization, but only 92 training songs caps how well it transfers to unseen music.
- **L1 magnitude loss is not a perceptual metric.** It shows the model trained and generalized; **SDR** (on reconstructed audio) is the real quality number and the right basis for the baseline-vs-SpecBandNet comparison.
- Expect **vocals/drums** to score better than **bass/other** - a known, explainable property of a small from-scratch model.

## Generated files

- `loss_comparison.png`
- `sdr_per_stem.png`
- `sdr_per_track.csv`
- `spectrogram_masks.png`
- `training_curves.png`
- `history.csv`
- `audio/  (mixture + 4 stems per track)`
- `metrics.csv`
