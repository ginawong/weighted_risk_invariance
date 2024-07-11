import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from domainbed import autoencoders


class EncoderWrapper(nn.Module):
    def __init__(self, enc, scale=1.0, offset=0.0):
        super().__init__()
        self.enc = enc
        self.scale = scale
        self.offset = offset

    def forward(self, x):
        return self.enc(x * self.scale + self.offset)


def train_mnist(train_splits, input_channels, feature_dims, device):
    model = autoencoders.mnist.MNISTAutoencoder(feature_dims, img_channels=input_channels).to(device)

    batch_size = 64
    SCALE = 2.0
    OFFSET = -1.0

    dataset = torch.utils.data.ConcatDataset([dataset for dataset, _ in train_splits])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.MSELoss()

    NUM_EPOCHS = 20
    LR = 4e-4
    WEIGHT_DECAY = 2e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print('Training featurizer')
    for epoch in range(NUM_EPOCHS):
        avg_std = 0
        avg_loss = 0
        num_samples = 0
        for data in loader:
            # mnist data is in [0,1] by default, want to give it approximately zero mean
            img = data[0].to(device) * SCALE + OFFSET
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item() * len(img)
            avg_std = avg_std + output.std(dim=0).mean().item() * len(img)
            num_samples += len(img)
        avg_std = avg_std / num_samples
        avg_loss = avg_loss / num_samples
        print(f'epoch {epoch + 1}/{NUM_EPOCHS}  --  avg_loss: {avg_loss:0.4f}  --  avg batch pixel stddev: {avg_std:0.4f}')

    model.eval()

    return EncoderWrapper(model.encoder, SCALE, OFFSET)
