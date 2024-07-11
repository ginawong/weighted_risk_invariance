import torch.nn as nn


class MNISTAutoencoder(nn.Module):
    def __init__(self, feature_dims, img_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 20 * 20, feature_dims),
            # nn.Softmax(dim=1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dims, 1600),
            nn.ReLU(),
            nn.Linear(1600, 12800),
            nn.ReLU(),
            nn.Unflatten(1, (32, 20, 20)),
            nn.ConvTranspose2d(32, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, img_channels, 5),
            nn.Tanh()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
