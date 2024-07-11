import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import resnet
from tqdm import tqdm


def get_model(arch, feature_dims, num_classes, pretrained):
    if arch == 'resnet18':
        model = resnet.resnet18(pretrained)
        conv2 = model.layer4[1].conv2
        model.layer4[1].conv2 = nn.Conv2d(conv2.in_channels, feature_dims, conv2.kernel_size, conv2.stride, conv2.padding, bias=False)
        model.layer4[1].bn2 = nn.BatchNorm2d(feature_dims)
        model.layer4[1].downsample = nn.Sequential(
            nn.Conv2d(model.layer4[1].conv1.in_channels, feature_dims, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dims))
        model.fc = nn.Linear(feature_dims, num_classes)
    elif arch == 'resnet50':
        model = resnet.resnet50(pretrained)
        conv3 = model.layer4[2].conv3
        model.layer4[2].conv3 = nn.Conv2d(conv3.in_channels, feature_dims, conv3.kernel_size, conv3.stride, conv3.padding, bias=False)
        model.layer4[2].bn3 = nn.BatchNorm2d(feature_dims)
        model.layer4[2].downsample = nn.Sequential(
            nn.Conv2d(model.layer4[2].conv1.in_channels, feature_dims, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dims))
        model.fc = nn.Linear(feature_dims, num_classes)
    else:
        raise ValueError(f"Unexpected arch {arch}")
    return model


def remove_classifier(model):
    model.fc = nn.Identity()
    return model


def _train(model, optimizer, dataset, device, num_batches, batch_size):
    from torch.utils.data import RandomSampler
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2,
                        sampler=RandomSampler(dataset, replacement=True, num_samples=num_batches * batch_size))

    criterion = nn.CrossEntropyLoss()

    avg_loss, avg_acc, num_samples, step = 0, 0, 0, 0

    pbar = tqdm(loader, total=len(loader), ncols=0, desc=f'Classifier Featurizer Train')
    postfix = dict()
    for data in pbar:
        img = data[0].to(device)
        label = data[1].to(device)

        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = torch.sum(torch.argmax(output, dim=1) == label).item() / len(img)

        avg_loss = avg_loss + loss.item() * len(img)
        avg_acc = avg_acc + acc * len(img)
        num_samples += len(img)

        postfix['loss'] = f'{avg_loss / num_samples:0.4g}'
        postfix['acc'] = f'{avg_acc / num_samples * 100:0.2f}%'
        pbar.set_postfix(postfix)

        if (step + 1) % 100 == 0 or step + 1 == len(loader):
            avg_loss = avg_loss / num_samples
            avg_acc = avg_acc / num_samples
            print(f'\nstep {step + 1}/{len(loader)}  --  avg_loss: {avg_loss:0.4f}  --  avg acc: {avg_acc * 100:0.2f}%')
            avg_loss, avg_acc, num_samples = 0, 0, 0

        step += 1


def train(train_splits, feature_dims, num_classes, device, pretrained=False, arch='resnet50', batch_size=64, steps=10000):
    model = get_model(arch, feature_dims, num_classes, pretrained)
    model.to(device)

    dataset = torch.utils.data.ConcatDataset([dataset for dataset, _ in train_splits])

    LR = 5e-5
    WEIGHT_DECAY = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    _train(model, optimizer, dataset, device, steps, batch_size)

    model.eval()

    remove_classifier(model)

    return model
