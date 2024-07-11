# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import List

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "CMNISTHetero25",
    "CMNISTHetero25_CovShift65",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug1D(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        from domainbed.sim_dataset import init_datasets
        inv_dims, spu_dims = 15, 15
        self.num_classes = 5
        self.input_shape = (inv_dims + spu_dims,)
        _, self.datasets = init_datasets(num_envs=5, r_dims=inv_dims, s_dims=spu_dims, num_classes=self.num_classes, samples_per_env=10000)
        for i, d in enumerate(self.datasets):
            d.name = f"Dbg_{i}"


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, env_names, dataset_transform, input_shape,
                 num_classes, idealized=False):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        self.idealized = idealized

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i], env_names[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams, idealized=False):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9], ColoredMNIST.ENVIRONMENTS,
                                           self.color_dataset, (2, 28, 28,), 2,
                                           idealized=idealized)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment, env_name):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        binary_labels = (labels < 5).float()
        # Flip label with probability 0.25
        binary_labels = self.torch_xor_(binary_labels,
                                        self.torch_bernoulli_(0.25, len(binary_labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(binary_labels,
                                 self.torch_bernoulli_(environment,
                                                       len(binary_labels)))
        if self.idealized:
            labels = labels.long()
            colors = colors.long()
            x = torch.cat([labels[:, None], colors[:, None]], dim=1)
            y = binary_labels.long()
            dataset = TensorDataset(x, y)
        else:
            images = torch.stack([images, images], dim=1)
            # Apply the color to the image by zeroing out the other color channel
            images[torch.tensor(range(len(images))), (
                1 - colors).long(), :, :] *= 0

            x = images.float().div_(255.0)
            y = binary_labels.view(-1).long()

            dataset = TensorDataset(x, y)

        dataset.name = env_name
        return dataset

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class CMNISTHeteroBase(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    EASY_HARD_LABELS = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 1,
        5: 0,
        6: 0,
        7: 1,
        8: 1,
        9: 1,
    }

    def __init__(self, root, test_envs, hparams, idealized=False):
        super(CMNISTHeteroBase, self).__init__(root, [0.1, 0.2, 0.9], ColoredMNIST.ENVIRONMENTS,
                                               self.color_dataset, (2, 28, 28,), 2,
                                               idealized=idealized)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.colors = None

    @property
    def flip_probs(self):
        """ override and return a two-tuple containing probability of flipping easy examples and hard examples """
        # return (easy_prob, hard_prob)
        raise NotImplementedError

    def color_dataset(self, images, labels, environment, env_name):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        binary_labels = (labels < 5).float()

        # Flip binary label with different probabilities
        for label, easy_hard in self.EASY_HARD_LABELS.items():
            mask = (labels == label)
            binary_labels[mask] = self.torch_xor_(binary_labels[mask],
                                                  self.torch_bernoulli_(self.flip_probs[easy_hard], sum(mask)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(binary_labels,
                                 self.torch_bernoulli_(environment,
                                                       len(binary_labels)))

        if self.idealized:
            labels = labels.long()
            colors = colors.long()
            x = torch.cat([labels[:, None], colors[:, None]], dim=1)
            y = binary_labels.long()
            dataset = TensorDataset(x, y)
        else:
            images = torch.stack([images, images], dim=1)
            # Apply the color to the image by zeroing out the other color channel
            images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

            x = images.float().div_(255.0)
            y = binary_labels.view(-1).long()

            dataset = TensorDataset(x, y)
        dataset.name = env_name
        return dataset

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class CMNISTHetero25(CMNISTHeteroBase):
    @property
    def flip_probs(self):
        return 0.05, 0.25


class CMNISTHeteroCovShiftBase(CMNISTHeteroBase):
    @property
    def easy_ratios(self):
        raise NotImplementedError

    def __init__(self, root, test_envs, hparams, idealized=False):
        super(MultipleEnvironmentMNIST, self).__init__()
        environments = [0.1, 0.2, 0.9]
        env_names = ColoredMNIST.ENVIRONMENTS
        dataset_transform = self.color_dataset
        input_shape = (2, 28, 28,)
        num_classes = 2
        self.idealized = idealized

        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        EASY_DIGITS = [k for k, v in self.EASY_HARD_LABELS.items() if v == 0]
        easy_mask = torch.from_numpy(np.isin(original_labels, EASY_DIGITS))
        hard_mask = torch.logical_not(easy_mask)

        # partition data by easy/hard digits
        original_images_easy = original_images[easy_mask]
        original_labels_easy = original_labels[easy_mask]
        original_images_hard = original_images[hard_mask]
        original_labels_hard = original_labels[hard_mask]
        num_easy = len(original_images_easy)
        num_hard = len(original_images_hard)
        num_total = len(original_images)
        num_envs = len(environments)
        samples_per_env = num_total // num_envs

        EASY_RATIOS = np.array(self.easy_ratios) / sum(self.easy_ratios)
        self.datasets = []

        # draw samples for each environment
        env_images: List[torch.Tensor] = []
        env_labels: List[torch.Tensor] = []
        idx0_easy, idx0_hard = 0, 0
        for i in range(num_envs):
            idx1_easy = idx0_easy + int(EASY_RATIOS[i] * num_easy) if i < num_envs - 1 else num_easy
            num_easy_env = idx1_easy - idx0_easy
            num_hard_env = samples_per_env - num_easy_env
            idx1_hard = idx0_hard + num_hard_env if i < num_envs - 1 else num_hard

            env_images.append(torch.cat([
                original_images_easy[idx0_easy:idx1_easy],
                original_images_hard[idx0_hard:idx1_hard]
            ]))
            env_labels.append(torch.cat([
                original_labels_easy[idx0_easy:idx1_easy],
                original_labels_hard[idx0_hard:idx1_hard]
            ]))

            # reshuffle
            shuffle = torch.randperm(len(env_images[-1]))
            env_images[-1] = env_images[-1][shuffle]
            env_labels[-1] = env_labels[-1][shuffle]

            idx0_easy, idx0_hard = idx1_easy, idx1_hard

        for i, (images, labels) in enumerate(zip(env_images, env_labels)):
            self.datasets.append(dataset_transform(images, labels, environments[i], env_names[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class CMNISTHetero25_CovShift65(CMNISTHeteroCovShiftBase):
    @property
    def easy_ratios(self):
        return 0.65, 0.05, 0.3

    @property
    def flip_probs(self):
        return 0.05, 0.25


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           RotatedMNIST.ENVIRONMENTS,
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle, env_name):
        if torchvision.__version__.startswith('0.8'):
            rotate_tform = lambda x: rotate(x, angle, fill=(0,), resample=Image.BILINEAR)
        else:
            rotate_tform = lambda x: rotate(x, angle, fill=(0,),
                                            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(rotate_tform),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        dataset = TensorDataset(x, y)
        dataset.name = env_name
        return dataset


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)
            env_dataset.name = self.ENVIRONMENTS[i]
            env_dataset.labels = env_dataset.samples

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

