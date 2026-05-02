import torchvision as tv
import numpy as np
from typing import Literal

from datasets import load_dataset



rng = np.random.default_rng(42)
train_inds = []
val_inds = []
test_inds = []
for n in range(38):
    _train_inds = rng.choice(800, 640, replace=False)
    _testval_inds = [j for j in range(800) if j not in _train_inds]
    _val_inds = rng.choice(_testval_inds, 80, replace=False)
    _test_inds = np.array([j for j in _testval_inds if j not in _val_inds])

    train_inds.extend(_train_inds + n*800)
    val_inds.extend(_val_inds + n*800)
    test_inds.extend(_test_inds + n*800)

debug_inds = rng.choice(800*38, 640) # for debug purposes

_patternnet_indices = {
    'train': train_inds,
    'test': val_inds,
    'val': test_inds,
    'debug': debug_inds
}


class PatternNet:
    """
    Aerial image dataset for multi-label classification.
    From Huggingface.
    """
    def __init__(
            self, 
            split: Literal['train', 'val', 'test', 'debug'], 
            transform=None,
            n_samples=None,
            sample_cut_ratio=None,
            sample_cut_seed=None
        ):
        """
        n_samples is only used in debug mode
        """
        self.ds = load_dataset('blanchon/PatternNet')['train'] # there is only train, so we split the dataset ourselves

        self.inds = _patternnet_indices[split]

        if split == 'debug':
            if n_samples is not None:
                self.inds = self.inds[:n_samples]
            if sample_cut_ratio is not None:
                raise ValueError('sample_cut_ratio is not supported for debug split. Please set n_samples to a smaller number instead.')
        else:
            if n_samples is not None:
                raise ValueError('n_samples is only supported for debug split. Please set sample_cut_ratio and sample_cut_seed instead.')
            if sample_cut_ratio is not None:
                assert sample_cut_seed is not None, 'sample_cut_seed must be provided if sample_cut_ratio is provided for non-debug splits'
                _sample_inds = np.random.default_rng(sample_cut_seed).choice(len(self.inds), int(len(self.inds) * sample_cut_ratio), replace=False)
                self.inds = [self.inds[i] for i in _sample_inds]
            # assert n_samples is None, f'cannot set n_samples for split={split}'

        self._N = len(self.inds)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: x

    def __getitem__(self, index: int):
        imglbl = self.ds[self.inds[index]]
        return self.transform(imglbl['image']), imglbl['label']

    def __len__(self):
        return self._N

