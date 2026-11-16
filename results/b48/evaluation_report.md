# Music Source Separation - Baseline U-Net: Evaluation Report

## Model & configuration

- Parameters: **17.46 M**
- Architecture: spectrogram U-Net, 4 softmax masks (stems sum to mixture)
- Sample rate / n_fft / hop: 44100 / 2048 / 512
- Segment length: 5.0s | Batch size: 6 | Base channels: 48
- Optimizer: Adam (lr 0.001), L1 magnitude loss, AMP=on

## Training summary

- Best checkpoint: epoch **48**, validation loss **0.11794**
- Epochs run: 50 | Total training time: ~595.4 min (~11.9 min/epoch)

## Final-model loss (L1 on magnitude spectrogram, lower = better)

| split | vocals | drums | bass | other | overall |
| --- | --- | --- | --- | --- | --- |
| train | 0.0970 | 0.1058 | 0.0539 | 0.1275 | 0.0961 |
| validation | 0.1149 | 0.1076 | 0.0788 | 0.1704 | 0.1179 |
| test | 0.0629 | 0.0853 | 0.0358 | 0.1008 | 0.0712 |

## Separation quality - SDR / SIR / SAR (dB, higher = better)

| stem | SDR | SIR | SAR |
| --- | --- | --- | --- |
| vocals | 6.08 | 13.63 | 7.01 |
| drums | 4.86 | 11.63 | 6.65 |
| bass | 2.08 | 4.99 | 6.70 |
| other | 2.18 | 7.29 | 4.94 |

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
