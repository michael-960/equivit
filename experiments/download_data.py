import torchvision as tv
import os

DATA_ROOT = os.environ.get('EXPERIMENT_DATA_DIR', None)

if DATA_ROOT is None:
    print('EXPERIMENT_DATA_DIR is not an environment variable')
    exit(1)

print(DATA_ROOT)

print('Downloading CIFAR10')

tv.datasets.CIFAR10(train=True, download=True, root=DATA_ROOT)
tv.datasets.CIFAR10(train=False, download=True, root=DATA_ROOT)


print('Downloading PCAM')

tv.datasets.PCAM(split='train', download=True, root=DATA_ROOT)
tv.datasets.PCAM(split='val', download=True, root=DATA_ROOT)
tv.datasets.PCAM(split='test', download=True, root=DATA_ROOT)

print('Downloading CIFAR100')

tv.datasets.CIFAR100(train=True, download=True, root=DATA_ROOT)
tv.datasets.CIFAR100(train=False, download=True, root=DATA_ROOT)


print('Downloading Food101')

tv.datasets.Food101(split='train', download=True, root=DATA_ROOT)
tv.datasets.Food101(split='test', download=True, root=DATA_ROOT)

