# model.py
# =============================================================================
# Spectrogram U-Net for source separation.
#
#   Input : mixture magnitude spectrogram   (B, 1, F, T)
#   Output: 4 soft masks in [0, 1]          (B, 4, F, T)
#
# The estimated stem magnitudes are simply:  mask_i * mixture_magnitude.
# A mask value near 1 means "this time-frequency bin belongs to source i",
# near 0 means "suppress this bin for source i".
#
# SHAPE HANDLING (important):
# Each MaxPool halves F and T, so after 4 pools the input must be divisible
# by 16. Real spectrograms are not (F = 1025, T = 259), so forward() zero-pads
# up to the next multiple of 16 and crops the output back at the end. This
# guarantees the encoder/decoder shapes ALWAYS line up at the skip
# connections - no crop helpers needed.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    """ENCODER unit: two rounds of (Conv2d -> BatchNorm -> LeakyReLU)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # bias=False because BatchNorm has its own learnable shift
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):


    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            # in = out_ch * 2 because of the channel-wise skip concatenation
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)                    # upsample: (B, C, F, T) -> (B, C, 2F, 2T)
        x = torch.cat([x, skip], dim=1)   # skip connection: concat on channel axis
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,in_channels=1,num_sources=config.NUM_SOURCES,base=config.BASE_CHANNELS):     
        super().__init__()

        # ----------------------- Encoder (downsampling path) -----------------
        # Spatial size after each level (relative to padded input):
        self.enc1 = ConvBlock(in_channels, base)      # F,    T      ch: 32
        self.enc2 = ConvBlock(base,       base * 2)   # F/2,  T/2    ch: 64
        self.enc3 = ConvBlock(base * 2,   base * 4)   # F/4,  T/4    ch: 128
        self.enc4 = ConvBlock(base * 4,   base * 8)   # F/8,  T/8    ch: 256
        self.pool = nn.MaxPool2d(2)                   # halves F and T

        # ----------------------- Bottleneck ----------------------------------
        self.bottleneck = ConvBlock(base * 8, base * 16)   # F/16, T/16  ch: 512

        # ----------------------- Decoder (upsampling path) -------------------
        self.up4 = UpBlock(base * 16, base * 8)
        self.up3 = UpBlock(base * 8,  base * 4)
        self.up2 = UpBlock(base * 4,  base * 2)
        self.up1 = UpBlock(base * 2,  base)

        # ----------------------- Output head ---------------------------------
        # 1x1 conv -> one logit map per source. The softmax that turns these
        # into masks lives in forward() so it can run ACROSS the source
        # dimension (dim=1), forcing the 4 masks to sum to exactly 1 at every
        # time-frequency bin. That makes the 4 estimated stems add back up to
        # the mixture -> no energy overshoot/clipping, and a clean property to
        # demonstrate (the separated stems re-sum to the original input).
        self.head = nn.Conv2d(base, num_sources, kernel_size=1)

    def forward(self, x):
        # ---- 1) pad F and T up to multiples of 16 ---------------------------
        _, _, f, t = x.shape
        pad_f = (16 - f % 16) % 16
        pad_t = (16 - t % 16) % 16
        # F.pad order for 4-D input: (T_left, T_right, F_top, F_bottom)
        x = F.pad(x, (0, pad_t, 0, pad_f))

        # ---- 2) encoder, keeping each level's output for the skips ----------
        s1 = self.enc1(x)               # full resolution
        s2 = self.enc2(self.pool(s1))   # 1/2
        s3 = self.enc3(self.pool(s2))   # 1/4
        s4 = self.enc4(self.pool(s3))   # 1/8

        # ---- 3) bottleneck ---------------------------------------------------
        b = self.bottleneck(self.pool(s4))   # 1/16

        # ---- 4) decoder with skip connections --------------------------------
        d = self.up4(b, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)

        # ---- 5) masks via softmax across sources, then crop the padding off --
        logits = self.head(d)                  # (B, 4, F_padded, T_padded)
        masks  = torch.softmax(logits, dim=1)  # sum over the 4 sources == 1
        return masks[:, :, :f, :t]             # back to the original (F, T)


if __name__ == "__main__":
    # Quick shape sanity check:  python model.py
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    dummy = torch.randn(2, 1, 1025, 259)     # (B, 1, F, T) like a real batch
    out = model(dummy)
    print(f"Params: {n_params:.2f} M | in: {tuple(dummy.shape)} -> out: {tuple(out.shape)}")
