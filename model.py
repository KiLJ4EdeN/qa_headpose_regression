import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # dw
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class HeadPoseMobileNetV2(nn.Module):
    def __init__(self, alpha=1.0, num_outputs=3):
        super(HeadPoseMobileNetV2, self).__init__()
        block = InvertedResidualBlock

        input_channel = _make_divisible(32 * alpha, 8)
        last_channel = _make_divisible(1280 * alpha, 8)

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First conv layer
        layers = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        # Inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = _make_divisible(c * alpha, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Final layers
        layers.extend([
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        ])

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.regressor(x)
        return x


if __name__ == "__main__":
    model = HeadPoseMobileNetV2(alpha=0.6, num_outputs=3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    dummy = torch.randn(4, 3, 64, 64)
    out = model(dummy)
    print("Output:", out.shape)  # Should be [4, 3]
