import torch
import torch.nn.functional as F
import torch.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tonic
from tonic import DiskCachedDataset
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from einops import repeat
import os


cls_num_classes = {
    "mnist": 10,
    "fashionmnist": 10,
    "caltech101": 101,
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
    "dvsgesture": 11,
    "dvscifar10": 10,
    "ncaltech101": 101,
    "ncars": 2,
    "nmnist": 10,
    "agnews": 4,
}


def dvs_channel_check_expend(x):
    """
    Ensure that the DVS dataset outputs two channels
    tensor -> tensor
    """
    if x.shape[1] == 1:
        return repeat(x, 'b c w h -> b (r c) w h', r=2)
    else:
        return x


def get_dvsgesture_data(
        root,
        batch_size,
        T,
        train=True,
        size=48,
        num_workers=8,
        event_transform=None,
        transform=None,
        target_transform=None,
        cache_path=None,
        num_copies=3):

    sensor_size = tonic.datasets.DVSGesture.sensor_size

    if train:
        if cache_path is None:
            path = os.path.join(root, 'DVS/DVSGesture/cache_train_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/DVSGesture/cache_train_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T),
            ])

        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: dvs_channel_check_expend(x),
                transforms.RandomCrop(size, padding=size // 12),
            ])

        train_dataset = tonic.datasets.DVSGesture(
            os.path.join(root, 'DVS/DVSGesture'),
            train=True, transform=event_transform, target_transform=target_transform)

        train_dataset = DiskCachedDataset(
            train_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=num_workers,
            shuffle=True)

        return train_loader

    else:
        if cache_path is None:
            path = os.path.join(root, 'DVS/DVSGesture/cache_test_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/DVSGesture/cache_test_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])

        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: dvs_channel_check_expend(x),
            ])

        test_dataset = tonic.datasets.DVSGesture(
            os.path.join(root, 'DVS/DVSGesture'),
            train=False, transform=event_transform, target_transform=target_transform)

        test_dataset = DiskCachedDataset(
            test_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=False,
            num_workers=num_workers,
            shuffle=False)

        return test_loader


def get_dvscifar10_data(
        root,
        batch_size,
        T,
        train=True,
        size=48,
        num_workers=8,
        event_transform=None,
        transform=None,
        target_transform=None,
        cache_path=None,
        radio=0.9,
        num_copies=3):

    """
    http://journal.frontiersin.org/article/10.3389/fnins.2017.00309/full
    """

    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size

    num_per_cls = 10000 // 10
    indices_train, indices_test = [], []
    for i in range(10):
        indices_train.extend(
            list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * radio))))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * radio), (i + 1) * num_per_cls)))

    if train:
        if cache_path is None:
            path = os.path.join(root, 'DVS/DVS_Cifar10/cache_train_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/DVS_Cifar10/cache_train_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])

        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: dvs_channel_check_expend(x),
                transforms.RandomCrop(size, padding=size // 12),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])

        train_dataset = tonic.datasets.CIFAR10DVS(
            os.path.join(root, 'DVS/DVS_Cifar10'),
            transform=event_transform, target_transform=target_transform)

        train_dataset = DiskCachedDataset(
            train_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
            pin_memory=True,
            drop_last=True,
            num_workers=num_workers)

        return train_loader

    else:
        if cache_path is None:
            path = os.path.join(root, 'DVS/DVS_Cifar10/cache_test_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/DVS_Cifar10/cache_test_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        event_transform = transforms.Compose([
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])

        transform = transforms.Compose([
            lambda x: torch.tensor(x, dtype=torch.float),
            lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
            lambda x: dvs_channel_check_expend(x)
        ])

        test_dataset = tonic.datasets.CIFAR10DVS(
            os.path.join(root, 'DVS/DVS_Cifar10'),
            transform=event_transform, target_transform=target_transform)

        test_dataset = DiskCachedDataset(
            test_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
            pin_memory=True,
            drop_last=False,
            num_workers=num_workers)

        return test_loader


def get_ncaltech101_data(
        root,
        batch_size,
        T,
        train=True,
        size=48,
        num_workers=8,
        event_transform=None,
        transform=None,
        target_transform=None,
        cache_path=None,
        radio=0.9,
        num_copies=3):
    """
    http://journal.frontiersin.org/Article/10.3389/fnins.2015.00437/abstract
    """
    sensor_size = tonic.datasets.NCALTECH101.sensor_size
    cls_count = tonic.datasets.NCALTECH101.cls_count
    dataset_length = tonic.datasets.NCALTECH101.length

    train_sample_weight = []
    train_sample_index = []
    train_count = 0
    test_sample_index = []
    idx_begin = 0
    for count in cls_count:
        sample_weight = dataset_length / count
        train_sample = round(radio * count)
        test_sample = count - train_sample
        train_count += train_sample
        train_sample_weight.extend(
            [sample_weight] * train_sample
        )
        train_sample_weight.extend(
            [0.] * test_sample
        )
        train_sample_index.extend(
            list((range(idx_begin, idx_begin + train_sample)))
        )
        test_sample_index.extend(
            list(range(idx_begin + train_sample, idx_begin + train_sample + test_sample))
        )
        idx_begin += count

    if train:
        if cache_path is None:
            path = os.path.join(root, 'DVS/NCALTECH101/cache_train_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/NCALTECH101/cache_train_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])

        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                transforms.RandomCrop(size, padding=size // 12),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])

        train_dataset = tonic.datasets.NCALTECH101(
            os.path.join(root, 'DVS/NCALTECH101'),
            transform=event_transform, target_transform=target_transform)

        train_dataset = DiskCachedDataset(
            train_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weight, train_count)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=num_workers)
        return train_loader

    else:
        if cache_path is None:
            path = os.path.join(root, 'DVS/NCALTECH101/cache_test_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/NCALTECH101/cache_test_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T), ])

        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
            ])

        test_dataset = tonic.datasets.NCALTECH101(
            os.path.join(root, 'DVS/NCALTECH101'),
            transform=event_transform, target_transform=target_transform)

        test_dataset = DiskCachedDataset(
            test_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_sample_index)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size,
            sampler=test_sampler,
            pin_memory=True, drop_last=False, num_workers=2
        )

        return test_loader


def get_ncars_data(
        root,
        batch_size,
        T,
        train=True,
        size=48,
        num_workers=8,
        event_transform=None,
        transform=None,
        target_transform=None,
        cache_path=None,
        num_copies=3):
    """
    https://ieeexplore.ieee.org/document/8578284/
    """
    sensor_size = tonic.datasets.NCARS.sensor_size

    if train:
        if cache_path is None:
            path = os.path.join(root, 'DVS/NCARS/cache_train_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/NCARS/cache_train_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])
        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: dvs_channel_check_expend(x),
                transforms.RandomCrop(size, padding=size // 12),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])

        train_dataset = tonic.datasets.NCARS(
            os.path.join(root, 'DVS/NCARS'),
            train=True, transform=event_transform, target_transform=target_transform)

        train_dataset = DiskCachedDataset(
            train_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            pin_memory=True, drop_last=True, num_workers=num_workers,
            shuffle=True,
        )

        return train_loader

    else:
        if cache_path is None:
            path = os.path.join(root, 'DVS/NCARS/cache_test_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/NCARS/cache_test_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])

        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: dvs_channel_check_expend(x),
            ])

        test_dataset = tonic.datasets.NCARS(
            os.path.join(root, 'DVS/NCARS'),
            train=False, transform=event_transform, target_transform=target_transform)

        test_dataset = DiskCachedDataset(
            test_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size,
            pin_memory=True, drop_last=False, num_workers=2,
            shuffle=False,
        )

        return test_loader


def get_nmnist_data(
        root,
        batch_size,
        T,
        train=True,
        size=48,
        num_workers=8,
        event_transform=None,
        transform=None,
        target_transform=None,
        cache_path=None,
        num_copies=3):

    sensor_size = tonic.datasets.NCARS.sensor_size

    if train:
        if cache_path is None:
            path = os.path.join(root, 'DVS/NMNIST/cache_train_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/NMNIST/cache_train_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])
        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: dvs_channel_check_expend(x),
                transforms.RandomCrop(size, padding=size // 12),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])

        train_dataset = tonic.datasets.NMNIST(
            os.path.join(root, 'DVS/NMNIST'),
            train=True, transform=event_transform, target_transform=target_transform)

        train_dataset = DiskCachedDataset(
            train_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            pin_memory=True, drop_last=True, num_workers=num_workers,
            shuffle=True,
        )

        return train_loader

    else:
        if cache_path is None:
            path = os.path.join(root, 'DVS/NMNIST/cache_test_%s_default' % T)
        elif type(cache_path) is str:
            path = os.path.join(root, 'DVS/NMNIST/cache_test_%s_%s' % (T, cache_path))
        else:
            TypeError('cache_path should be type string')

        if event_transform is None:
            event_transform = transforms.Compose([
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)])

        if transform is None:
            transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: dvs_channel_check_expend(x),
            ])

        test_dataset = tonic.datasets.NMNIST(
            os.path.join(root, 'DVS/NMNIST'),
            train=False, transform=event_transform, target_transform=target_transform)

        test_dataset = DiskCachedDataset(
            test_dataset,
            cache_path=path,
            transform=transform,
            target_transform=target_transform,
            num_copies=num_copies)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size,
            pin_memory=True, drop_last=False, num_workers=2,
            shuffle=False,
        )

        return test_loader


def get_mnist_data(
        root,
        batch_size,
        train=True,
        num_workers=8,
        transform=None,
        target_transform=None):

    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    if train:
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
            ])
        mnist_train = datasets.MNIST(root=root, train=True, download=True,
                                     transform=transform, target_transform=target_transform)

        train_iter = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True)

        return train_iter

    else:
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
            ])
        mnist_test = datasets.MNIST(
            root=root, train=False, download=True,
            transform=transform, target_transform=target_transform)

        test_iter = torch.utils.data.DataLoader(
            mnist_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True)

        return test_iter


def get_fashionmnist_data(
        root,
        batch_size,
        train=True,
        num_workers=8,
        transform=None,
        target_transform=None):

    if train:
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor()
            ])

        fashionmnist_train = datasets.FashionMNIST(
            root=root, train=True, download=True,
            transform=transform, target_transform=target_transform)

        train_iter = torch.utils.data.DataLoader(
            fashionmnist_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True)

        return train_iter

    else:
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        fashionmnist_test = datasets.FashionMNIST(
            root=root, train=False, download=True,
            transform=transform, target_transform=target_transform)

        test_iter = torch.utils.data.DataLoader(
            fashionmnist_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True)

        return test_iter


def get_caltech101_data(
        root,
        batch_size,
        train=True,
        num_workers=8,
        transform=None,
        target_transform=None):

    if train:
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor()
            ])

        caltech101_train = datasets.Caltech101(
            root=root, train=True, download=True,
            transform=transform, target_transform=target_transform)

        train_iter = torch.utils.data.DataLoader(
            caltech101_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True)

        return train_iter

    else:
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        caltech101_test = datasets.Caltech101(
            root=root, train=False, download=True,
            transform=transform, target_transform=target_transform)

        test_iter = torch.utils.data.DataLoader(
            caltech101_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True)

        return test_iter


def get_cifar10_data(
        root,
        batch_size,
        train=True,
        num_workers=8,
        transform=None,
        target_transform=None,
        distributed=False):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if train:
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                normalize
            ])
        cifar10_train = datasets.CIFAR10(
            root=root, train=True, download=True,
            transform=transform, target_transform=target_transform)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(cifar10_train)
            train_iter = torch.utils.data.DataLoader(
                cifar10_train,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True,
                sampler=train_sampler)
        else:
            train_iter = torch.utils.data.DataLoader(
                cifar10_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True)
        return train_iter

    else:
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
        cifar10_test = datasets.CIFAR10(
            root=root, train=False, download=True,
            transform=transform, target_transform=target_transform)

        if distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                cifar10_test, shuffle=False, drop_last=True)

            test_iter = torch.utils.data.DataLoader(
                cifar10_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                sampler=val_sampler)
        else:
            test_iter = torch.utils.data.DataLoader(
                cifar10_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True)

        return test_iter


def get_cifar100_data(
        root,
        batch_size,
        train=True,
        num_workers=8,
        transform=None,
        target_transform=None,
        distributed=False):
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    if train:
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                normalize
            ])
        cifar100_train = datasets.CIFAR100(
            root=root, train=True, download=True,
            transform=transform, target_transform=target_transform)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(cifar100_train)

            train_iter = torch.utils.data.DataLoader(
                cifar100_train,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True,
                sampler=train_sampler)
        else:
            train_iter = torch.utils.data.DataLoader(
                cifar100_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True)
        return train_iter

    else:
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), normalize])

        cifar100_test = datasets.CIFAR100(
            root=root, train=False, download=True,
            transform=transform, target_transform=target_transform)

        if distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                cifar100_test, shuffle=False, drop_last=True)

            test_iter = torch.utils.data.DataLoader(
                cifar100_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                sampler=val_sampler)
        else:
            test_iter = torch.utils.data.DataLoader(
                cifar100_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True)

        return test_iter


def get_imagenet2_data(
        root,
        batch_size,
        train=True,
        num_workers=8,
        transform=None,
        distributed=False):
    '''
    load imagenet 2012
    we use images in train/ for training, and use images in val/ for testing
    https://github.com/pytorch/examples/tree/master/imagenet
    '''
    traindir = os.path.join(root, 'ILSVRC2012/train')
    valdir = os.path.join(root, 'ILSVRC2012/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if train:
        if transform is None:
            transform = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # ImageNetPolicy(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                # Cutout(n_holes=1, length=16),
                normalize
            ])

        train_dataset = datasets.ImageFolder(
            traindir, transform=transform)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True,
                sampler=train_sampler)

        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True)

        return train_loader

    else:
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

        val_dataset = datasets.ImageFolder(valdir, transform=transform)

        if distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                sampler=val_sampler)
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True)

        return val_loader

def get_imagenet_data(
        root,
        batch_size,
        train=True,
        num_workers=8,
        transform=None,
        distributed=False):

    traindir = os.path.join(root, 'ILSVRC2012/train')
    valdir = os.path.join(root, 'ILSVRC2012/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # ImageNetPolicy(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=val_sampler)

    return train_loader if train else val_loader


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img