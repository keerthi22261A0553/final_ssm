import torch
import torch.nn as nn


class UIBBlock(nn.Module):

    # Universal Inverted Bottleneck (SDD ST-01):
    # expansion -> depthwise convolution -> BatchNorm/SiLU -> projection

    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio=4,
        kernel_size=3,
        stride=(1, 1),
    ):

        super().__init__()

        # Expand from the wider side so block 1 (1 input channel)
        # still gets a useful hidden width
        hidden_channels = max(in_channels, out_channels) * expand_ratio

        padding = kernel_size // 2

        # 1x1 expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # kxk depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # 1x1 projection (no activation)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        stride = (stride, stride) if isinstance(stride, int) else stride

        # Residual only when input and output shapes match
        self.use_residual = (
            tuple(stride) == (1, 1)
            and in_channels == out_channels
        )

    def forward(self, x):

        out = self.project(self.depthwise(self.expand(x)))

        if self.use_residual:
            out = out + x

        return out


class UIBFeatureExtractor(nn.Module):

    # Four stacked UIB blocks (SDD LG-02, UIB Block 1-4)

    def __init__(
        self,
        in_channels=1,
        channels=(16, 24, 32, 48),
        strides=((2, 1), (2, 1), (2, 1), (2, 1)),
        expand_ratio=4,
        kernel_size=3,
    ):

        super().__init__()

        if len(channels) != 4 or len(strides) != 4:
            raise ValueError("Design specifies exactly 4 UIB blocks.")

        blocks = []

        for out_channels, stride in zip(channels, strides):

            blocks.append(
                UIBBlock(
                    in_channels,
                    out_channels,
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )

            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        self.out_channels = channels[-1]

    def forward(self, x):

        # Input : (B, 1, F, T)  spectrogram
        # Output: (B, C, F', T) feature map
        return self.blocks(x)


if __name__ == "__main__":

    model = UIBFeatureExtractor()

    # Batch of spectrograms from RadarSegmentDataset
    x = torch.randn(32, 1, 128, 33)

    print("Input shape :", tuple(x.shape))

    out = x

    for index, block in enumerate(model.blocks, start=1):

        out = block(out)

        print(
            f"UIB Block {index} : {tuple(out.shape)}"
            f"  residual={block.use_residual}"
        )

    # Shape as expected? (Figure A-2)
    assert out.shape == (32, 48, 8, 33), out.shape

    params = sum(p.numel() for p in model.parameters())

    print("Output shape:", tuple(model(x).shape))
    print("Parameters  :", f"{params:,}")
