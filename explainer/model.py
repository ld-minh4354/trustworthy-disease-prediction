import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(p=0.2)

        # For skip connection
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

    def forward(self, x):
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNet, self).__init__()
        self.layer1 = ResidualBlock(input_dim, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.layer4 = ResidualBlock(256, 512)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out