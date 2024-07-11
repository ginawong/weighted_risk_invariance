import os
import torch
import torch.nn as nn
import torch.nn.functional as nnf
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


def train_mnist(train_splits, input_channels, feature_dims, device, *, args):
    model = autoencoders.mnist.MNISTAutoencoder(feature_dims, img_channels=input_channels).to(device)

    batch_size = 64
    SCALE = 2.0
    OFFSET = -1.0

    featurizer_cache_fname = None
    arg_str = None
    if args is not None:
        latent_flag = f'dims={feature_dims}'
        args_test_env = ','.join(str(test_env) for test_env in args.test_envs)
        arg_str = f'{args.dataset}_bs={batch_size}_{latent_flag}_testenvs={args_test_env}'
        featurizer_cache_fname = f'classifier_featurizer_{arg_str}.pt'

    if featurizer_cache_fname is not None and os.path.exists(featurizer_cache_fname):
        print('Loading pretrained featurizer')
        model.load_state_dict(torch.load(featurizer_cache_fname, map_location=device))
        model.eval()
        return EncoderWrapper(model.encoder, SCALE, OFFSET), arg_str

    dataset = torch.utils.data.ConcatDataset([dataset for dataset, _ in train_splits])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.MSELoss()

    NUM_EPOCHS = 20
    LR = 4e-4
    WEIGHT_DECAY = 2e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    from tqdm import tqdm
    print('Training featurizer')
    for epoch in range(NUM_EPOCHS):
        avg_std = 0
        avg_loss = 0
        num_samples = 0
        for data in tqdm(loader, total=len(loader), ncols=0, desc=f'Featurizer Train {epoch+1}/{NUM_EPOCHS}'):
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

    if featurizer_cache_fname is not None:
        print('Saving pretrained featurizer')
        if os.path.dirname(featurizer_cache_fname):
            os.makedirs(os.path.dirname(featurizer_cache_fname), exist_ok=True)
        torch.save(model.state_dict(), featurizer_cache_fname)

    return EncoderWrapper(model.encoder, SCALE, OFFSET), arg_str


def _train_resnet(model, optimizer, dataset, device, num_epochs, lr, batch_size=64, debug_output_prefix='',
                  use_ssim=False, input_width=224):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if use_ssim:
        from domainbed.autoencoders import ssim
        objective = ssim.SSIM()
        objective.to(device)
        criterion = lambda x, y: -objective(x, y)
    else:
        criterion = nn.MSELoss()

    # normalization values applied in dataset
    data_std_orig = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    data_mean_orig = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    data_scale = data_std_orig
    data_shift = data_mean_orig

    from tqdm import tqdm
    for epoch in range(num_epochs):
        lr = adjust_learning_rate_cosine(optimizer, lr, epoch, num_epochs)

        avg_loss, avg_std, num_samples = 0, 0, 0

        pbar = tqdm(loader, total=len(loader), ncols=0, desc=f'Featurizer Train {epoch + 1}/{num_epochs}')
        postfix = dict()
        for data in pbar:
            img = (data[0] * data_scale + data_shift).to(device)
            if input_width != img.shape[2]:
                img = nnf.interpolate(img, size=(input_width, input_width), mode='bilinear', align_corners=False)

            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item() * len(img)
            avg_std = avg_std + output.std(dim=0).mean().item() * len(img)
            num_samples += len(img)

            postfix['lr'] = f'{lr:0.4g}'
            postfix['loss'] = f'{avg_loss / num_samples:0.4f}'
            pbar.set_postfix(postfix)

        avg_std = avg_std / num_samples
        avg_loss = avg_loss / num_samples
        print(f'epoch {epoch + 1}/{num_epochs}  --  avg_loss: {avg_loss:0.4f}  --  avg batch pixel stddev: {avg_std:0.4f}')


def train_224_model(train_splits, feature_dims, device, pretrained=False, arch='resnet18', disable_finetune=False,
                    batch_size=64, use_latent_img=False, use_ssim=False, args=None):
    model_input_width = 224
    if arch == 'resnet18':
        configs, bottleneck = autoencoders.resnet.get_configs('resnet18')
        model = autoencoders.resnet.ResNetAutoEncoder(feature_dims, configs, bottleneck, use_tanh=False,
                                                      use_latent_img=use_latent_img).to(device)
        pretrained_name = 'caltech256-resnet18.pth'
    elif arch == 'resnet50':
        configs, bottleneck = autoencoders.resnet.get_configs('resnet50')
        model = autoencoders.resnet.ResNetAutoEncoder(feature_dims, configs, bottleneck, use_tanh=False,
                                                      use_latent_img=use_latent_img).to(device)
        pretrained_name = 'caltech256-resnet50.pth'
    elif arch == 'vgg16':
        configs = autoencoders.vgg.get_configs('vgg16')
        model = autoencoders.vgg.VGGAutoEncoder(feature_dims, configs,
                                                use_latent_img=use_latent_img).to(device)
        pretrained_name = 'imagenet-vgg16.pth'
    elif arch == 'custom':
        import domainbed.autoencoders.custom
        model = autoencoders.custom.CustomAutoEncoder(feature_dims).to(device)
        model_input_width = model.input_width
        pretrained_name = 'does_not_exist'
    else:
        # Download other weights from https://github.com/Horizon2333/imagenet-autoencoder if needed
        raise ValueError("Unexpected arch")

    featurizer_cache_fname = None
    if args is not None:
        latent_flag = f'latent=img_ch={feature_dims}' if use_latent_img else f'latent=fc_dims={feature_dims}'
        ftflag = 'withft_' if (not disable_finetune and pretrained) else ''
        args_test_env = ','.join(str(test_env) for test_env in args.test_envs)
        arg_str = f'{args.dataset}_{arch}_bs={batch_size}_{latent_flag}_pt={int(pretrained)}_{ftflag}testenvs={args_test_env}'
        featurizer_cache_fname = f'featurizer_{arg_str}.pt'

    if featurizer_cache_fname is not None and os.path.exists(featurizer_cache_fname):
        model.load_state_dict(torch.load(featurizer_cache_fname, map_location=device))
        model.eval()
        return model.encoder

    dataset = torch.utils.data.ConcatDataset([dataset for dataset, _ in train_splits])

    # fine-tune before full training if pretrained==True
    FT_NUM_EPOCHS = 10
    FT_LR = 0.05
    FT_WEIGHT_DECAY = 1e-4
    FT_MOMENTUM = 0.9

    NUM_EPOCHS = 20
    LR = 0.05
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9

    if pretrained:
        model.load_pretrained(os.path.join(os.path.dirname(__file__), 'pretrained', pretrained_name))

        # finetune newly initialized parts of model unless told not to
        if not use_latent_img and not disable_finetune:
            dbg_img_prefix = None
            if args is not None:
                os.makedirs('images', exist_ok=True)
                dbg_img_prefix = f'images/img_{arg_str}_ft_ep_{{epoch:03d}}'
            optimizer = torch.optim.SGD(model.non_pretrained_parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY, momentum=FT_MOMENTUM)
            for p in model.pretrained_parameters():
                p.requires_grad_(False)
            _train_resnet(model, optimizer, dataset, device, FT_NUM_EPOCHS, FT_LR, batch_size=batch_size,
                          debug_output_prefix=dbg_img_prefix, use_ssim=use_ssim,
                          input_width=model_input_width)
            for p in model.pretrained_parameters():
                p.requires_grad_(True)

    dbg_img_prefix = None
    if args is not None:
        os.makedirs('images', exist_ok=True)
        dbg_img_prefix = f'images/img_{arg_str}_full_ep_{{epoch:03d}}'
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    _train_resnet(model, optimizer, dataset, device, NUM_EPOCHS, FT_LR, batch_size=batch_size,
                  debug_output_prefix=dbg_img_prefix, use_ssim=use_ssim,
                  input_width=model_input_width)

    model.eval()

    if featurizer_cache_fname is not None:
        print('Saving pretrained featurizer')
        if os.path.dirname(featurizer_cache_fname):
            os.makedirs(os.path.dirname(featurizer_cache_fname), exist_ok=True)
        torch.save(model.state_dict(), featurizer_cache_fname)

    return model.encoder


def adjust_learning_rate_cosine(optimizer, base_lr, current_epoch, total_epochs):
    """cosine learning rate annealing without restart"""
    import math
    lr = base_lr * 0.5 * (1. + math.cos(math.pi * current_epoch / total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
