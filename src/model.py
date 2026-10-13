import torch
import torch.nn as nn

from ssm import LightweightSSM, SequenceConverter
from uib import UIBFeatureExtractor


class UIBSSMClassifier(nn.Module):

    # Full forward path (SDD 1.4):
    # UIB Block 1-4 -> Feature-to-Sequence -> Lightweight SSM
    # (RMSNorm / Dropout) -> Pooling -> Fully Connected -> logits

    def __init__(
        self,
        num_classes=4,
        input_shape=(1, 128, 33),
        uib_channels=(16, 24, 32, 48),
        d_model=128,
        d_state=16,
        num_ssm_layers=2,
        dropout=0.1,
    ):

        super().__init__()

        self.input_shape = tuple(input_shape)

        self.uib = UIBFeatureExtractor(
            in_channels=input_shape[0],
            channels=uib_channels,
        )

        # Frequency bins left after the UIB blocks
        with torch.no_grad():
            freq_bins = self.uib(torch.zeros(1, *input_shape)).shape[2]

        self.sequence = SequenceConverter(
            in_channels=self.uib.out_channels,
            freq_bins=freq_bins,
            d_model=d_model,
        )

        self.ssm = LightweightSSM(
            d_model=d_model,
            d_state=d_state,
            num_layers=num_ssm_layers,
            dropout=dropout,
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):

        # x: (B, 1, F, T) spectrogram
        features = self.uib(x)

        # (B, N, D)
        sequence = self.sequence(features)

        sequence = self.ssm(sequence)

        # Global average pooling over the sequence
        pooled = sequence.mean(dim=1)

        # Logits only; CrossEntropyLoss applies Softmax internally
        return self.classifier(pooled)

    @torch.no_grad()
    def predict(self, x):

        # Inference (SDD IF-003): class index and probability vector
        self.eval()

        probabilities = torch.softmax(self.forward(x), dim=1)

        return probabilities.argmax(dim=1), probabilities


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    from dataset import RadarSegmentDataset

    torch.manual_seed(0)

    model = UIBSSMClassifier()

    # Real batch from the validation split
    dataset = RadarSegmentDataset("val")
    inputs, labels = next(iter(DataLoader(dataset, batch_size=32, shuffle=True)))

    logits = model(inputs)

    print("Input shape  :", tuple(inputs.shape))
    print("Logits shape :", tuple(logits.shape))

    assert logits.shape == (32, 4)

    # One training step works end to end
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    print("Initial loss :", f"{loss.item():.3f}  (random guess = ln 4 = 1.386)")

    predicted, probabilities = model.predict(inputs)

    print("Predicted    :", predicted[:8].tolist())
    print("Prob. sums   :", probabilities.sum(dim=1)[:4].tolist())

    print("\nParameters per module:")

    for name in ("uib", "sequence", "ssm", "classifier"):
        print(f"  {name:10s}: {count_parameters(getattr(model, name)):>8,}")

    total = count_parameters(model)

    print(f"  {'total':10s}: {total:>8,}  ({total * 4 / 1024:.0f} KB float32)")
