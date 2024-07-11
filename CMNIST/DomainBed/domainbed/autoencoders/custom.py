import torch
import torch.nn as nn


class CustomAutoEncoder(nn.Module):
    def __init__(self, feature_dims, latent_img=False):
        super().__init__()

        encoder_block_configs = [
            {'num_layers': 1, 'in_channels': 3, 'hidden_channels': 32, 'out_channels': 64, 'downsample': 'pool'},
            {'num_layers': 2, 'in_channels': 64, 'hidden_channels': 64, 'out_channels': 128, 'downsample': 'conv'},
            {'num_layers': 2, 'in_channels': 128, 'hidden_channels': 128,
             'out_channels': 32 if not latent_img else feature_dims,
             'downsample': 'conv'},
        ]
        encoder_hidden = 2 * feature_dims

        decoder_hidden = 2 * feature_dims
        decoder_block_configs = []
        for encoder_block_config in encoder_block_configs[::-1]:
            decoder_block_configs.append({
                'num_layers': encoder_block_config['num_layers'],
                'in_channels': encoder_block_config['out_channels'],
                'hidden_channels': encoder_block_config['hidden_channels'],
                'out_channels': encoder_block_config['in_channels'],
                'upsample': 'conv_transpose'
            })

        self.encoder = Encoder(encoder_block_configs, encoder_hidden, feature_dims, latent_img)
        self.decoder = Decoder(decoder_block_configs, feature_dims, decoder_hidden, latent_img)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    @property
    def input_width(self):
        # expected input shape
        return 56


class Encoder(nn.Module):
    def __init__(self, block_configs, hidden_dims, output_dims, latent_img):
        super().__init__()
        self.blocks = nn.Sequential(*[EncoderBlock(**config) for config in block_configs])
        if latent_img:
            self.linear = lambda x: x
        else:
            self.linear = nn.Sequential(*[
                nn.Linear(7 * 7 * block_configs[-1]['out_channels'], hidden_dims),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims, output_dims)
            ])

    def forward(self, x):
        x = self.blocks(x).flatten(start_dim=1)
        return self.linear(x)


class EncoderBlock(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, downsample):
        super().__init__()
        from_channels = in_channels
        layers = []
        for layer_idx in range(num_layers):
            to_channels = out_channels if layer_idx == num_layers - 1 else hidden_channels
            layers.append(EncoderLayer(from_channels, to_channels, downsample))
            downsample = None
            from_channels = to_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_method=None):
        super().__init__()

        assert downsample_method in {None, 'conv', 'pool'}

        stride = 2 if downsample_method == 'conv' else 1

        modules = [
           nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True)
        ]
        if downsample_method == 'pool':
            modules.append(nn.MaxPool2d(2, stride=2))

        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, block_configs, input_dims, hidden_dims, latent_img=False):
        super().__init__()
        if latent_img:
            self.liner = lambda x: x
        else:
            self.linear = nn.Sequential(*[
                nn.Linear(input_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims, block_configs[0]['in_channels'] * 7 * 7),
                nn.ReLU(inplace=True),
            ])

        blocks = []
        to_final = len(block_configs)
        for block_config in block_configs:
            blocks.append(DecoderBlock(**block_config, final=(to_final == 1)))
            to_final -= 1
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.linear(x).reshape(x.shape[0], -1, 7, 7)
        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, upsample, final=False):
        super().__init__()
        from_channels = in_channels
        layers = []
        for layer_idx in range(num_layers):
            to_channels = out_channels if layer_idx == num_layers - 1 else hidden_channels
            layers.append(DecoderLayer(from_channels, to_channels, upsample, final=final and (layer_idx == num_layers - 1)))
            upsample = None
            from_channels = to_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_method=None, final=False):
        super().__init__()

        assert upsample_method in {None, 'conv_transpose', 'upsample'}

        modules = []
        if upsample_method == 'conv_transpose':
            modules.append(nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
        elif upsample_method == 'upsample':
            modules.append(nn.Upsample(scale_factor=2))

        if upsample_method != 'conv_transpose':
            modules.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))

        if final:
            modules.append(nn.Sigmoid())
        else:
            modules.extend([
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        return self.layer(x)
