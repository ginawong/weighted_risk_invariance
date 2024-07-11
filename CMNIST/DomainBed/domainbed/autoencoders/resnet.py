import torch.nn as nn
import torch
import copy

"""
ResNetAutoEncoder from https://github.com/Horizon2333/imagenet-autoencoder/
"""


def get_configs(arch='resnet50'):

    # True or False refers to whether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")


class ResNetAutoEncoder(nn.Module):

    def __init__(self, feature_dims, configs, bottleneck, use_tanh=False, use_latent_img=False):
        super(ResNetAutoEncoder, self).__init__()
        self.use_latent_img = use_latent_img
        self.feature_dims = feature_dims
        self.configs = copy.deepcopy(configs)
        self.bottleneck = bottleneck
        self.encoder = ResNetEncoder(feature_dims, configs=configs, bottleneck=bottleneck,
                                     use_latent_img=use_latent_img)
        self.decoder = ResNetDecoder(feature_dims, configs=configs[::-1], bottleneck=bottleneck, use_tanh=use_tanh,
                                     use_latent_img=use_latent_img)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def load_pretrained(self, filename):
        device = next(self.parameters()).device
        state_dict = torch.load(filename, map_location=device)['state_dict']
        state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        if self.use_latent_img:
            # remove weights that have changed size
            my_state_dict = self.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in my_state_dict and my_state_dict[k].shape == v.shape}
            self.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            assert len(unexpected_keys) == 0
            if missing_keys:
                allowed_missing = {'encoder.fc.weight', 'encoder.fc.bias',
                                   'decoder.fc.weight', 'decoder.fc.bias',
                                   'decoder.upsample.weight'}
                missing_keys = set(missing_keys) - allowed_missing
                assert len(missing_keys) == 0

    def non_pretrained_parameters(self):
        non_pretrained = []
        whitelist = [self.decoder]
        if self.use_latent_img:
            whitelist.append(self.encoder.conv5)
        else:
            whitelist.append(self.encoder.fc)

        for module in whitelist:
            for p in module.parameters():
                non_pretrained.append(p)
        return non_pretrained

    def pretrained_parameters(self):
        blacklist = {'fc'}
        if self.use_latent_img:
            blacklist.add('conv5')
        pretrained = []
        for name, module in self.encoder.named_modules():
            if name not in blacklist:
                for p in module.parameters():
                    pretrained.append(p)
        return pretrained


class ResNetEncoder(nn.Module):

    def __init__(self, feature_dims, configs, bottleneck=False, use_latent_img=False):
        super(ResNetEncoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configured")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:
            expansion = 4
            self.conv2 = EncoderBottleneckBlock(in_channels=64, hidden_channels=64, up_channels=256,
                                                layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderBottleneckBlock(in_channels=256, hidden_channels=128, up_channels=512,
                                                layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderBottleneckBlock(in_channels=512, hidden_channels=256, up_channels=1024,
                                                layers=configs[2], downsample_method="conv")
        else:
            expansion = 1
            self.conv2 = EncoderResidualBlock(in_channels=64, hidden_channels=64,
                                              layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderResidualBlock(in_channels=64, hidden_channels=128,
                                              layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256,
                                              layers=configs[2], downsample_method="conv")

        self.use_latent_img = use_latent_img
        if use_latent_img:
            if bottleneck:
                self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048,
                                                    out_channels=feature_dims, layers=configs[3],
                                                    downsample_method="conv")
            else:
                self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, out_channels=feature_dims,
                                                  layers=configs[3], downsample_method="conv")
        else:
            if bottleneck:
                self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048,
                                                    layers=configs[3], downsample_method="conv")
            else:
                self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512,
                                                  layers=configs[3], downsample_method="conv")
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * expansion, feature_dims)


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if not self.use_latent_img:
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)

        return x


class ResNetDecoder(nn.Module):

    def __init__(self, feature_dims, configs, bottleneck=False, use_tanh=False, use_latent_img=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configured")

        self.use_latent_img = use_latent_img
        if use_latent_img:
            if bottleneck:
                self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024,
                                                    layers=configs[0], special_in_channels=feature_dims)
            else:
                self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0],
                                                  in_channels=feature_dims)
        else:
            expansion = 4 if bottleneck else 1
            self.fc = nn.Linear(feature_dims, 128)
            self.upsample = nn.ConvTranspose2d(128, 512 * expansion, 3, dilation=3, bias=False)
            if bottleneck:
                self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024,
                                                    layers=configs[0])
            else:
                self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])

        if bottleneck:
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,
                                                layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512, hidden_channels=128, down_channels=256,
                                                layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256, hidden_channels=64, down_channels=64,
                                                layers=configs[3])
        else:
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64, layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64, output_channels=64, layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1,
                               bias=False),
        )

        if use_tanh:
            self.gate = nn.Tanh()
        else:
            self.gate = nn.Sigmoid()

    def forward(self, x):
        if not self.use_latent_img:
            x = torch.relu(self.fc(x))
            x = self.upsample(x[..., None, None])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x


class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv", out_channels=None):
        super(EncoderResidualBlock, self).__init__()
        if out_channels is None:
            out_channels = hidden_channels

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 out_channels=None if layers > 1 else out_channels,
                                                 downsample=True)
                elif i == layers - 1:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels,
                                                 out_channels=out_channels, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels,
                                                 downsample=False)

                self.add_module('%02d EncoderLayer' % i, layer)

        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 out_channels=None if layers > 1 else out_channels,
                                                 downsample=False)
                elif i == layers - 1:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels,
                                                 out_channels=out_channels, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels,
                                                 downsample=False)

                self.add_module('%02d EncoderLayer' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method="conv", out_channels=None):
        super(EncoderBottleneckBlock, self).__init__()

        if out_channels is None:
            out_channels = up_channels

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels if layers > 1 else out_channels, downsample=True)
                elif i == layers - 1:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels,
                                                   up_channels=out_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels, downsample=False)

                self.add_module('%02d EncoderLayer' % i, layer)

        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels if layers > 1 else out_channels,
                                                   downsample=False)
                elif i == layers - 1:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels,
                                                   up_channels=out_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels, downsample=False)

                self.add_module('%02d EncoderLayer' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers, in_channels=None):
        super(DecoderResidualBlock, self).__init__()

        if in_channels is None:
            in_channels = hidden_channels

        for i in range(layers):

            if i == 0:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels,
                                             output_channels=hidden_channels if layers > 1 else output_channels,
                                             upsample=layers == 1, in_channels=in_channels)
            elif i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels,
                                             upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels,
                                             upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers, special_in_channels=None):
        super(DecoderBottleneckBlock, self).__init__()

        if special_in_channels is None:
            special_in_channels = in_channels

        for i in range(layers):

            if i == 0:
                layer = DecoderBottleneckLayer(in_channels=special_in_channels, hidden_channels=hidden_channels,
                                               down_channels=in_channels if layers > 1 else down_channels,
                                               upsample=(layers == 1))
            elif i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=in_channels, upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample, out_channels=None):
        super(EncoderResidualLayer, self).__init__()

        if out_channels is None:
            out_channels = hidden_channels
        self.out_channels = out_channels

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
        elif out_channels != in_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x


class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x


class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample, in_channels=None):
        super(DecoderResidualLayer, self).__init__()

        if in_channels is None:
            in_channels = hidden_channels

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False)
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1,
                          bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=output_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        elif in_channels != output_channels:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=1, stride=1,
                          padding=0, bias=False)
            )
        else:
            self.upsample = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x


class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0,
                          bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0,
                          bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x

if __name__ == "__main__":
    import os
    model = ResNetAutoEncoder(512, *get_configs('resnet18'), use_tanh=False)
    model.load_pretrained(os.path.join(os.path.dirname(__file__), 'pretrained/caltech256-resnet18.pth'))