import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceConverter(nn.Module):

    # Feature-to-Sequence (SDD IF-002):
    # (B, C, F, T) feature map -> (B, N=T, D) sequence

    def __init__(self, in_channels, freq_bins, d_model):

        super().__init__()

        self.projection = nn.Linear(in_channels * freq_bins, d_model)

    def forward(self, x):

        batch, channels, freq, time = x.shape

        # Each time frame becomes one sequence element
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)

        return self.projection(x)


class DiagonalSSM(nn.Module):

    # Diagonal state-space model (S4D-style, SDD AL-01):
    #   h[k] = dA * h[k-1] + dB * u[k]
    #   y[k] = C * h[k] + D * u[k]
    # Computed as a causal convolution with kernel K[k] = C * dA^k * dB
    # (time-domain conv1d; equivalent to the recurrence below)

    def __init__(self, d_model, d_state=16, dt_min=0.001, dt_max=0.1):

        super().__init__()

        # Per-channel step size, log-uniform in [dt_min, dt_max]
        log_dt = (
            torch.rand(d_model)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )

        self.log_dt = nn.Parameter(log_dt)

        # A = -(1, 2, ..., N) keeps every state stable (S4D-Real)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)

        self.A_log = nn.Parameter(torch.log(A).repeat(d_model, 1))

        self.B = nn.Parameter(torch.ones(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        # Precomputed kernel for deployment (see freeze); not saved
        self.register_buffer("frozen_kernel", None, persistent=False)

    @torch.no_grad()
    def freeze(self, length):

        # Inference/export only: fix the kernel for a known sequence
        # length so the SSM becomes a plain depthwise convolution
        self.frozen_kernel = self.kernel(length)

    def discretize(self):

        dt = torch.exp(self.log_dt).unsqueeze(-1)
        A = -torch.exp(self.A_log)

        # Zero-order hold
        dA = torch.exp(dt * A)
        dB = (dA - 1) / A * self.B

        return dA, dB

    def kernel(self, length):

        dA, dB = self.discretize()

        steps = torch.arange(length, device=dA.device)

        # (H, N, L): dA^k for k = 0 .. L-1
        powers = dA.unsqueeze(-1) ** steps

        # (H, L)
        return torch.einsum("hn,hnl->hl", self.C * dB, powers)

    def forward(self, u):

        # u: (B, L, H)
        batch, length, _ = u.shape

        if self.frozen_kernel is not None and self.frozen_kernel.shape[-1] == length:
            K = self.frozen_kernel
        else:
            K = self.kernel(length)

        u_t = u.transpose(1, 2)

        # Causal depthwise convolution y[t] = sum_j K[j] * u[t - j].
        # Short sequences (N = 33) make this cheap, and unlike FFT it
        # exports to ONNX and can be quantized for edge deployment.
        y = F.conv1d(
            F.pad(u_t, (length - 1, 0)),
            K.flip(-1).unsqueeze(1),
            groups=K.shape[0],
        )

        y = y + u_t * self.D.unsqueeze(-1)

        return y.transpose(1, 2)

    def forward_recurrent(self, u):

        # Step-by-step recurrence, used to verify forward()
        dA, dB = self.discretize()

        batch, length, d_model = u.shape

        h = u.new_zeros(batch, d_model, dA.shape[-1])

        outputs = []

        for k in range(length):

            h = dA * h + dB * u[:, k].unsqueeze(-1)

            outputs.append((h * self.C).sum(-1) + self.D * u[:, k])

        return torch.stack(outputs, dim=1)


class LightweightSSMLayer(nn.Module):

    # SDD ST-02: sequence -> state processing -> RMSNorm -> Dropout

    def __init__(self, d_model, d_state=16, dropout=0.1):

        super().__init__()

        self.ssm = DiagonalSSM(d_model, d_state)

        # Diagonal SSM has no channel mixing; add it here
        self.activation = nn.SiLU()
        self.output = nn.Linear(d_model, d_model)

        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        y = self.output(self.activation(self.ssm(x)))

        x = self.norm(x + y)

        return self.dropout(x)


class LightweightSSM(nn.Module):

    def __init__(self, d_model=128, d_state=16, num_layers=2, dropout=0.1):

        super().__init__()

        self.layers = nn.Sequential(*[
            LightweightSSMLayer(d_model, d_state, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):

        # Input / output: (B, N, D)
        return self.layers(x)


if __name__ == "__main__":

    torch.manual_seed(0)

    # UIB output shape from uib.py
    features = torch.randn(32, 48, 8, 33)

    converter = SequenceConverter(in_channels=48, freq_bins=8, d_model=128)
    ssm = LightweightSSM(d_model=128)

    sequence = converter(features)
    output = ssm(sequence)

    print("UIB feature map :", tuple(features.shape))
    print("Sequence B×N×D  :", tuple(sequence.shape))
    print("SSM output      :", tuple(output.shape))

    assert sequence.shape == (32, 33, 128)
    assert output.shape == (32, 33, 128)

    # Convolution must equal the state-space recurrence
    layer = ssm.layers[0].ssm

    with torch.no_grad():
        max_diff = (layer(sequence) - layer.forward_recurrent(sequence)).abs().max()

    print("Conv vs recurrence max diff:", f"{max_diff.item():.2e}")

    assert max_diff < 1e-4

    # Gradients reach every parameter
    output.sum().backward()

    missing = [
        name for name, p in ssm.named_parameters()
        if p.grad is None
    ]

    print("Parameters without gradient:", missing or "none")

    print("Converter parameters:", f"{sum(p.numel() for p in converter.parameters()):,}")
    print("SSM parameters      :", f"{sum(p.numel() for p in ssm.parameters()):,}")
