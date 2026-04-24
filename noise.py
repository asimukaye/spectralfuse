import torch
from torch import Tensor
from torch.utils.data import Subset, Dataset
import numpy as np
from typing import Sequence, Protocol
import random


class MappedDataset(Protocol):
    """Dataset with the indices mapped to the original dataset"""

    @property
    def targets(self) -> list: ...
    @property
    def class_to_idx(self) -> dict: ...
    @property
    def indices(self) -> Sequence[int]: ...

    def __len__(self) -> int: ...


def check_for_mapping(dataset: Dataset) -> MappedDataset:
    if not hasattr(dataset, "class_to_idx"):
        raise TypeError(f"Dataset {dataset} does not have class_to_idx")
    if not hasattr(dataset, "targets"):
        raise TypeError(f"Dataset {dataset} does not have targets")
    if not hasattr(dataset, "indices"):
        raise TypeError(f"Dataset {dataset} does not have indices")
    return dataset  # type: ignore


def extract_root_dataset(subset: Subset) -> Dataset:
    if isinstance(subset.dataset, Subset):
        return extract_root_dataset(subset.dataset)
    else:
        assert isinstance(subset.dataset, Dataset), "Unknown subset nesting"
        return subset.dataset


def extract_root_dataset_and_indices(
    subset: Subset, indices=None
) -> tuple[Dataset, np.ndarray]:
    # ic(type(subset.indices))
    if indices is None:
        indices = subset.indices
    np_indices = np.array(indices)
    if isinstance(subset.dataset, Subset):
        # ic(type(subset.dataset))
        mapped_indices = np.array(subset.dataset.indices)[np_indices]
        # ic(mapped_indices)
        return extract_root_dataset_and_indices(subset.dataset, mapped_indices)
    else:
        assert isinstance(subset.dataset, Dataset), "Unknown subset nesting"
    
        return subset.dataset, np_indices



class LabelNoiseSubset(Subset):
    """Wrapper of `torch.utils.Subset` module for label flipping."""

    def __init__(self, dataset: Dataset, flip_pct: float, index_label_mapping: dict, split_map:list):
        
        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
            self.indices = dataset.indices
            root_dataset, mapped_ids = extract_root_dataset_and_indices(dataset)
            checked_dataset = check_for_mapping(root_dataset)
        else:
            self.dataset = dataset
            checked_dataset = check_for_mapping(dataset)
            self.indices = checked_dataset.indices
            mapped_ids = np.array(checked_dataset.indices)

        self.subset = self._flip_set(dataset, checked_dataset, mapped_ids, flip_pct)  # type: ignore

    def _flip_set(
        self,
        subset: Subset,
        dataset: MappedDataset,
        mapped_ids: np.ndarray,
        flip_pct: float,
    ) -> Subset:
        total_size = len(subset)
        # dataset, mapped_ids = extract_root_dataset_and_indices(subset)
        # dataset = check_for_mapping(dataset)

        samples = np.random.choice(
            total_size, size=int(flip_pct * total_size), replace=False
        )

        selected_indices = mapped_ids[samples]
        # ic(samples, selected_indices)
        class_ids = list(dataset.class_to_idx.values())
        for idx, dataset_idx in zip(samples, selected_indices):
            _, lbl = subset[idx]
            assert lbl == dataset.targets[dataset_idx]
            # ic(lbl, )
            excluded_labels = [cid for cid in class_ids if cid != lbl]
            # changed_label = np.random.choice(excluded_labels)
            # ic(changed_label)
            dataset.targets[dataset_idx] = np.random.choice(excluded_labels)
            # print('\n')
        return subset

    def __getitem__(self, index):
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        return len(self.subset)

    def __repr__(self):
        return f"{repr(self.subset.dataset)}_LabelFlipped"


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )

def add_gaussian_noise(tensor: Tensor, mean=0.0, std=1.0) -> Tensor:
    noisy_image = tensor + torch.randn(tensor.size()) * std + mean
    # return torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image

class NoisySubset(Subset):
    """Wrapper of `torch.utils.Subset` module for applying individual transform."""

    def __init__(self, subset: Subset, mean: float, std: float, pct_noisy: float):
        # def __init__(self, subset: Subset,  mean:float, std: float):
        self.dataset = subset.dataset
        self.indices = subset.indices
        # self.noise = AddGaussianNoise(mean, std)
        # self._subset = subset
        self._subset = []
        subset_size = len(subset)
        # local_indices = [range(subset_size)]
        # assert subset_size == len(self.indices)
        num_noisy = int(pct_noisy * subset_size)
        # print(num_noisy, pct_noisy, subset_size)
        if num_noisy > 0:
            noise_indices = np.random.choice(
                subset_size, size=num_noisy, replace=False
            )
        else:
            noise_indices = []
        # self.noise_indices = noise_indices

        # print(len(noise_indices))
        for lidx, gidx in zip(range(subset_size), self.indices):
            inputs, targets = self.dataset[gidx]
            # print(inputs.mean(), inputs.std())
            # print(lidx, gidx)
            if lidx in noise_indices:
                self._subset.append((add_gaussian_noise(inputs, mean, std), targets))
            else:
                self._subset.append((inputs, targets))

    def __getitem__(self, idx):
        # inputs, targets = self._subset[idx]
        return self._subset[idx]
        # return self.noise(inputs), targets

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"{repr(self.dataset)}_GaussianNoise"


class BlackWhiteSubset(Subset):
    """Wrapper of `torch.utils.Subset` module for applying individual transform."""

    def __init__(self, subset: Subset, pct_noisy: float):
        # def __init__(self, subset: Subset,  mean:float, std: float):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self._subset = []
        subset_size = len(subset)
        num_noisy = int(pct_noisy * subset_size)
 
        if num_noisy > 0:
            noise_indices = np.random.choice(
                subset_size, size=num_noisy, replace=False
            )
        else:
            noise_indices = []
        # self.noise_indices = noise_indices

        for lidx, gidx in zip(range(subset_size), self.indices):
            input, target = self.dataset[gidx]

            if lidx in noise_indices:
                b_w = random.choice([torch.max(input), torch.min(input)])
                # print(b_w)
                newinputs = torch.ones_like(input)*b_w
                self._subset.append((newinputs, target))
            else:
                self._subset.append((input, target))

    def __getitem__(self, idx):
        return self._subset[idx]

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"{repr(self.dataset)}_BlackWhite"