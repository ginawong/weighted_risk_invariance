import torch
import torch.nn as nn


def get_configs(arch='vgg16'):
    if arch == 'vgg11':
        configs = [1, 1, 2, 2, 2]
    elif arch == 'vgg13':
        configs = [2, 2, 2, 2, 2]
    elif arch == 'vgg16':
        configs = [2, 2, 3, 3, 3]
    elif arch == 'vgg19':
        configs = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Undefined model")

    return configs


class VGGAutoEncoder(nn.Module):

    def __init__(self, feature_dims, configs, use_latent_img=False):
        super(VGGAutoEncoder, self).__init__()

        # VGG without Bn as AutoEncoder is hard to train
        self.use_latent_img = use_latent_img
        self.feature_dims = feature_dims
        self.encoder = VGGEncoder(feature_dims=feature_dims, configs=configs, enable_bn=True,
                                  use_latent_img=use_latent_img)
        self.decoder = VGGDecoder(feature_dims=feature_dims, configs=configs[::-1], enable_bn=True,
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
            # model surgery to match desired number of channels in latent img, keep pretrained kernels with largest
            # encoder norm

            if self.feature_dims != 512:
                ec5_conv_key = 'encoder.conv5.2 EncoderLayer.layer.0'
                ec5_bn_key = 'encoder.conv5.2 EncoderLayer.layer.1'
                dc1_conv_key = 'decoder.conv1.0 UpSampling'
                keys = {f'{ec5_conv_key}.weight',
                        f'{ec5_conv_key}.bias',
                        f'{ec5_bn_key}.weight',
                        f'{ec5_bn_key}.bias',
                        f'{ec5_bn_key}.running_mean',
                        f'{ec5_bn_key}.running_var',
                        f'{dc1_conv_key}.weight'}

                kernel_indices = torch.randperm(state_dict[next(iter(keys))].shape[0], device=device)[:self.feature_dims]
                with torch.no_grad():
                    my_state_dict = self.state_dict()
                    for key in keys:
                        param_pt = state_dict[key]
                        param_new = torch.clone(my_state_dict[key])
                        param_new[:len(kernel_indices)] = param_pt[kernel_indices]
                        state_dict[key] = param_new

            self.load_state_dict(state_dict)
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
        whitelist = [self.decoder]
        if self.use_latent_img:
            whitelist.append(self.encoder.conv5)
        else:
            whitelist.append(self.encoder.fc)

        non_pretrained = []
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


class VGG(nn.Module):

    def __init__(self, configs, num_classes=1000, img_size=224, enable_bn=False):
        super(VGG, self).__init__()

        self.encoder = VGGEncoder(configs=configs, enable_bn=enable_bn)

        self.img_size = img_size / 32

        self.fc = nn.Sequential(
            nn.Linear(in_features=int(self.img_size * self.img_size * 512), out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.encoder(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class VGGEncoder(nn.Module):

    def __init__(self, feature_dims, configs, enable_bn=False, use_latent_img=False):
        super(VGGEncoder, self).__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")

        self.use_latent_img = use_latent_img

        self.conv1 = EncoderBlock(input_dim=3, output_dim=64, hidden_dim=64, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = EncoderBlock(input_dim=64, output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        if use_latent_img:
            self.conv5 = EncoderBlock(input_dim=512, output_dim=feature_dims, hidden_dim=512, layers=configs[4],
                                      enable_bn=enable_bn)
        else:
            self.conv5 = EncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4],
                                      enable_bn=enable_bn)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, feature_dims)



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


class VGGDecoder(nn.Module):

    def __init__(self, feature_dims, configs, enable_bn=False, use_latent_img=False):
        super(VGGDecoder, self).__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")

        self.use_latent_img = use_latent_img
        if not use_latent_img:
            self.fc = nn.Linear(feature_dims, 128)
            self.upsample = nn.ConvTranspose2d(128, 512, 3, dilation=3, bias=False)
            self.conv1 = DecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[0],
                                      enable_bn=enable_bn)
        else:
            self.conv1 = DecoderBlock(input_dim=feature_dims, output_dim=512, hidden_dim=512, layers=configs[0],
                                      enable_bn=enable_bn)
        self.conv2 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = DecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = DecoderBlock(input_dim=64, output_dim=3, hidden_dim=64, layers=configs[4], enable_bn=enable_bn)
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


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(EncoderBlock, self).__init__()

        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d EncoderLayer' % i, layer)

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):
                if i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d DecoderLayer' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class EncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)


class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):

        return self.layer(x)


if __name__ == "__main__":
    input = torch.randn((5, 3, 224, 224))

    configs = get_configs()

    model = VGGAutoEncoder(configs)

    output = model(input)

    print(output.shape)



