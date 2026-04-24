from pathlib import Path
import torch
import logging
from dataclasses import dataclass
import torchvision
import os
import json
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import numpy as np

from torchvision import transforms
from torchvision.transforms import functional as TF

# from torchvision.transforms import v2
from torchvision.datasets import CIFAR10, VisionDataset, CIFAR100, MNIST, FashionMNIST, EMNIST
# from torchtext.datasets import SST2
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# dataset wrapper module
from configdefs import DATA_PATH

from splits import (
    get_free_rider_split,
    get_step_label_skew_split_v2,
    get_label_skew_only_split_v2,
    get_step_quantity_split,
    get_iid_split_v3,
    get_dirichlet_split,
    add_feature_noise_to_datasets,
    add_label_noise_to_datasets
)
from models import get_clf_model


def get_index_label_mapping(dataset, root:Path, dataset_name="CIFAR10"):
    """Generates a mapping of indices to labels for the dataset."""
    if os.path.exists(root / f"{dataset_name.lower()}_index_label_mapping.json"):
        with open(root / f"{dataset_name.lower()}_index_label_mapping.json", "r") as f:
            index_label_mapping = json.load(f)
    else:
        index_label_mapping = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            index_label_mapping[label].append(idx)
        with open(root / f"{dataset_name.lower()}_index_label_mapping.json", "w") as f:
            json.dump(index_label_mapping, f)
    return index_label_mapping


def get_client_train_val_indices(clients_indices, train_ratio=0.9):
    client_train_val_indices = {}
    for i, c_idx in enumerate(clients_indices):
        train_indices = np.random.choice(
            c_idx, int(train_ratio * len(c_idx)), replace=False
        )
        val_indices = np.setdiff1d(c_idx, train_indices)
        client_train_val_indices[i] = (train_indices, val_indices)
    return client_train_val_indices



class VisionClfDataset(Dataset):
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.targets = self.dataset.targets  # type: ignore
        self.indices = np.arange(len(self.dataset))  # type: ignore
        self.class_to_idx = dataset.class_to_idx

    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        return inputs, targets

    def __len__(self):
        return len(self.dataset)  # type: ignore


# RFFL version
class FastCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        from torch import from_numpy

        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)

        self.targets = torch.Tensor(self.targets).long()

        # https://github.com/kuangliu/pytorch-cifar/issues/16
        # https://github.com/kuangliu/pytorch-cifar/issues/8
        for i, (mean, std) in enumerate(
            zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ):
            self.data[:, i].sub_(mean).div_(std)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print(
            "CIFAR10 data shape {}, targets shape {}".format(
                self.data.shape, self.targets.shape
            )
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class FastCIFAR100(CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        from torch import from_numpy

        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)

        self.targets = torch.Tensor(self.targets).long()

        # https://github.com/kuangliu/pytorch-cifar/issues/16
        # https://github.com/kuangliu/pytorch-cifar/issues/8
        for i, (mean, std) in enumerate(
            zip((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ):
            self.data[:, i].sub_(mean).div_(std)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print(
            "CIFAR10 data shape {}, targets shape {}".format(
                self.data.shape, self.targets.shape
            )
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def get_cifar10(root=DATA_PATH):
    num_classes = 10
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train = CIFAR10(root=root, train=True, download=True, transform=train_transforms)
    test = CIFAR10(root=root, train=False, download=True, transform=test_transforms)
    index_label_mapping = get_index_label_mapping(train, root, "CIFAR10")
    metadata = {"in_channels": 3,
                "num_classes": num_classes,
                "index_label_mapping": index_label_mapping}
    return train, test, metadata

    # return VisionClfDataset(train, "CIFAR10"), VisionClfDataset(test, "CIFAR10")


def get_emnist(root=DATA_PATH):

    train_transforms = transforms.Compose(
        [
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    num_classes = 47

    train = EMNIST(
        root=DATA_PATH,
        split="balanced",
        train=True,
        download=True,
        transform=train_transforms,
    )
    test = EMNIST(
        root=DATA_PATH,
        split="balanced",
        train=False,
        download=True,
        transform=test_transforms,
    )

    index_label_mapping = get_index_label_mapping(train, root, "EMNIST")

    metadata = {"in_channels": 3,
                "num_classes": num_classes,
                "index_label_mapping": index_label_mapping}

    return train, test, metadata


def get_femnist(root=DATA_PATH, nclients=200, custom_seed=42,  test_dist='intra-client', pooled_train= False):
    from datasets import concatenate_datasets
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(28, padding=2),
            transforms.Pad(2, fill=255, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize((0.9722,), (0.1365,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Pad(2, fill=255, padding_mode='constant'),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0.9722,), (0.1365,)),
        ]
    )
    ## Changing the custom seed will change the client selection and data splits

    global_seed = int(os.environ.get("PYTHONHASHSEED", 42))
    num_classes = 62
    # 1) FEMNIST, partitioned by writer_id (each partition == one writer)
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
        shuffle=False)

    part = fds.partitioners["train"]
    total_clients = part.num_partitions
    # sample 200 clients for faster testing
    # client_ids = list(range(num_clients))
    rng = np.random.default_rng(custom_seed)
    if nclients == -1:
        nclients = total_clients
    client_ids = list(rng.choice(range(total_clients), nclients, replace=False))

    # 2) For each writer, split 90/10 locally (uniform random within that writer)
    client_train = {}
    client_test = {}
    chosen_test = {}

    def ds_with_transform(ds, transform):
        # works with Hugging Face Dataset objects
        def _apply(batch):
            imgs = batch["image"]
            
            batch["image"] = [transform(img) for img in imgs]
            batch["label"] = batch["character"]
            # labels = batch["character"]

            del batch["character"]
            # del batch["writer_id"]
            # del batch["hsf_id"]
            # print("Transformed batch size:", len(batch["image"]))
            # return batch['image'], labels
            return batch
        ds.set_format("torch", columns=["image", "character"])
        return ds.with_transform(_apply)
    # .with_format("torch")
    

    # for pid in range(num_clients):
    for cid in client_ids:
        ds = fds.load_partition(cid, split="train")  # this writer's samples only
        # Make sure every writer contributes something: use an int count for test_size
        n_test = max(1, int(round(0.10 * len(ds))))
        split = ds.train_test_split(test_size=n_test, seed=custom_seed)
        # client_train[cid] = split["train"]
        # client_train[cid] = split["train"].with_transform(to_tensor).with_format("torch")
        if pooled_train:
            client_train[cid] = split["train"]
        else:
            client_train[cid] = ds_with_transform(split["train"], train_transform)

        client_test[cid] = split["test"]

    # 3) (Optional) Build a single centralized test set from all writers' 10%
    if test_dist == 'random-client':
        # Load the official FEMNIST test set (unseen writers)
        rng = np.random.default_rng(global_seed)
        test_ids = list(rng.choice(range(total_clients), nclients, replace=False))
        for cid in test_ids:
            ds = fds.load_partition(cid, split="train")  # this writer's samples only

            n_test = max(1, int(round(0.10 * len(ds))))
            split = ds.train_test_split(test_size=n_test, seed=global_seed)
            chosen_test[cid] = split["test"]

        central_test = concatenate_datasets([chosen_test[cid] for cid in test_ids])
    elif test_dist == 'intra-client':
        central_test = concatenate_datasets(list(client_test.values()))
    elif test_dist == 'cross-client':
        rng = np.random.default_rng(global_seed)
        cross_clients = list(set(range(total_clients)) - set(client_ids))
        test_ids = list(rng.choice(cross_clients, nclients, replace=False))
        for cid in test_ids:
            ds = fds.load_partition(cid, split="train")  # this writer's samples only

            n_test = max(1, int(round(0.10 * len(ds))))
            split = ds.train_test_split(test_size=n_test, seed=global_seed)
            chosen_test[cid] = split["test"]

        central_test = concatenate_datasets([chosen_test[cid] for cid in test_ids])
    else:
        raise ValueError(f"Unknown test_dist option: {test_dist}")

    central_test = ds_with_transform(central_test, test_transform)

    metadata = {"in_channels": 1,
                "num_classes": num_classes}
    if pooled_train:
        all_train = concatenate_datasets(list(client_train.values()))
        all_train = ds_with_transform(all_train, train_transform)
        return all_train, central_test, metadata
    else:
        return list(client_train.values()), central_test, metadata


def get_fast_cifar10():
    train = FastCIFAR10(root=DATA_PATH, train=True, download=True)
    test = FastCIFAR10(root=DATA_PATH, train=False, download=True)
    metadata = {"in_channels": 3,
                "num_classes": 10,
                "index_label_mapping": None}
    return train, test, metadata


def get_cifar100(root=DATA_PATH):

    num_classes = 100
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train = CIFAR100(root=root, train=True, download=True, transform=train_transform)
    test = CIFAR100(root=root, train=False, download=True, transform=test_transform)
    index_label_mapping = get_index_label_mapping(train, root, "CIFAR100")
    metadata = {"in_channels": 3,
                "num_classes": num_classes,
                "index_label_mapping": index_label_mapping}
    return train, test, metadata


def get_fedisic():
    from flamby.datasets.fed_isic2019 import FedIsic2019

    if os.uname().nodename == "gigabyte-W771-Z00-00":
        fedisic_root = "/share/"
    else:
        fedisic_root = os.environ["BIG_DATA_PATH"] + '/fedisic' # "/home/asim.ukaye/fed_learning/flbase/data/"
    client_sets = []
    for i in range(6):
        client_sets.append(
            FedIsic2019(
                center=i,
                train=True,
                pooled=False,
                data_path=fedisic_root,
            )
        )
    testset = FedIsic2019(train=False, pooled=True, data_path=fedisic_root)
    num_classes = int(8)
    in_channels = 3
    metadata = {"in_channels": in_channels,
                "num_classes": num_classes}

    return client_sets, testset, metadata

def get_mnist(root=DATA_PATH):
    num_classes = 10
    in_channels = 1
    train_transforms = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.Pad(2, fill=0, padding_mode='constant'),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train = MNIST(root=root, train=True, download=True, transform=train_transforms)
    test = MNIST(root=root, train=False, download=True, transform=test_transforms)

    index_label_mapping = get_index_label_mapping(train, root, "MNIST")

    metadata = {"in_channels": in_channels,
                "num_classes": num_classes,
                "index_label_mapping": index_label_mapping}

    return train, test, metadata

def get_fashionmnist(root=DATA_PATH):
    num_classes = 10
    in_channels = 1
    train_transforms = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,)),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,)),
        ]
    )
    train = FashionMNIST(root=root, train=True, download=True, transform=train_transforms)
    test = FashionMNIST(root=root, train=False, download=True, transform=test_transforms)
    index_label_mapping = get_index_label_mapping(train, root, "FashionMNIST")

    metadata = {"in_channels": in_channels,
                "num_classes": num_classes,
                "index_label_mapping": index_label_mapping}
    return train, test, metadata


def get_fedisic_pooled():
    pass

# def get_sst(root=DATA_PATH):
#     num_classes = 2
#     train = SST2(split='train')
#     test = SST2(split='test')
#     index_label_mapping = defaultdict(list)
#     if os.path.exists(root / "sst_index_label_mapping.json"):
#         with open(root / "sst_index_label_mapping.json", "r") as f:
#             index_label_mapping = json.load(f)
#     else:
#         index_label_mapping = get_index_label_mapping(train)
#         with open(root / "sst_index_label_mapping.json", "w") as f:
#             json.dump(index_label_mapping, f)


def subsample_dataset(dataset: Dataset, fraction: float):
    return Subset(dataset, np.random.randint(0, len(dataset) - 1, int(fraction * len(dataset))))  # type: ignore



def get_standard_clf_datasets(name)->tuple[Dataset, Dataset, dict]: 
    match name:
        case "cifar10":
            return get_cifar10()
        case "fastcifar10":
            return get_fast_cifar10()
        case "cifar100":
            return get_cifar100()
        case "emnist":
            return get_emnist()
        case "mnist":
            return get_mnist()
        case "fashionmnist":
            return get_fashionmnist()
        case _:
            raise NotImplementedError(f"Dataset {name} not implemented in standard clf datasets.")

def get_natural_split_dataset(dataset_name: str):
    match dataset_name:
        case "fedisic":
            return get_fedisic()
        case "femnist":
            return get_femnist()
        case _:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented in natural split datasets.")

def get_simulated_split_dataset(
    dataset_name: str,
    split_name: str,
    num_clients: int,
    split_config: dict = {},
    subsample=1.0,
):
    train, test, metadata = get_standard_clf_datasets(dataset_name)

    # print(f"Dataset {dataset_name} has {num_classes} classes, index keys: {index_label_mapping.keys()}.")
    if subsample < 1.0:
        train = subsample_dataset(train, subsample)
        test = subsample_dataset(test, subsample)

    split_config = split_config or {}
    client_data_indices = []
    match split_name:
        case "iid":
            client_data_indices = get_iid_split_v3(
                num_clients=num_clients, index_label_mapping=metadata["index_label_mapping"]
            )

        case "step_label_skew":
            if not split_config:
                # min_labels = split_config["min_labels"]
                if dataset_name == "cifar10":
                    split_config["min_labels"] = 2
                elif dataset_name == "cifar100":
                    split_config["min_labels"] = 20
                elif dataset_name == "mnist" or dataset_name == "fashionmnist":
                    split_config["min_labels"] = 2
                elif dataset_name == "emnist":
                    split_config["min_labels"] = 10
                else:
                    raise NotImplementedError(
                        f"Dataset {dataset_name} not tested for step_label_skew split."
                    )
            client_data_indices = get_step_label_skew_split_v2(
                num_clients=num_clients,
                num_classes=metadata["num_classes"],
                index_label_mapping=metadata["index_label_mapping"],
                **split_config,
            )
        case "only_label_skew":
            if not split_config:
                if dataset_name == "cifar10":
                    split_config["min_labels"] = 2
                    split_config["max_samples_per_label"] = 2000
                    split_config["min_samples_per_label"] = 100
                elif dataset_name == "cifar100":
                    split_config["min_labels"] = 20
                    split_config["max_samples_per_label"] = 400
                    split_config["min_samples_per_label"] = 100
                elif dataset_name == "mnist" or dataset_name == "fashionmnist":
                    split_config["min_labels"] = 2
                    split_config["max_samples_per_label"] = 3000
                    split_config["min_samples_per_label"] = 100
                elif dataset_name == "emnist":
                    split_config["min_labels"] = 10
                    split_config["max_samples_per_label"] = 2000
                    split_config["min_samples_per_label"] = 100
                else:
                    raise NotImplementedError(
                        f"Dataset {dataset_name} not tested for only_label_skew split."
                    )

            client_data_indices = get_label_skew_only_split_v2(
                num_clients=num_clients,
                num_classes=metadata["num_classes"],
                index_label_mapping=metadata["index_label_mapping"],
                **split_config,
            )
        case "step_quantity":
            client_data_indices = get_step_quantity_split(
                num_clients=num_clients,
                num_classes=metadata["num_classes"],
                index_label_mapping=metadata["index_label_mapping"],
            )
        case "dirichlet":
            if not split_config:
                split_config["alpha"] = 1.0

            client_data_indices = get_dirichlet_split(
                index_label_mapping=metadata["index_label_mapping"],
                num_clients=num_clients,
                alpha=split_config["alpha"],
                num_classes=metadata["num_classes"],
            )
        case "free_rider":
            client_data_indices = get_free_rider_split(
                index_label_mapping=metadata["index_label_mapping"],
                num_clients=num_clients,
                num_classes=metadata["num_classes"],
                free_rider_idx=split_config.get("free_rider_idx", 0),
                free_rider_actual_size=split_config.get("free_rider_actual_size", 10),
            )
        case _:
            raise NotImplementedError(f"Split {split_name} not implemented.")

    print(
        "Len of datasets:", {i: len(client_data_indices[i]) for i in range(num_clients)}
    )

    # Save the split config
    if split_config:
        split_config_path = Path(os.environ["OUT_DIR"]) / f"split_config.json"
        with open(split_config_path, "w") as f:
            json.dump(split_config, f, indent=4)

    client_sets = []
    for i, c_idx in enumerate(client_data_indices):
        client_sets.append(Subset(train, c_idx))  # type: ignore

    meta = metadata.copy()
    del meta["index_label_mapping"]
    return client_sets, test, meta


def get_fl_datasets_and_model(
    dataset_name: str, split_name: str, model_name: str, num_clients: int, split_config:dict = {}, subsample=1.0
):
    
    if split_name == "natural":
        client_sets, testset, metadata = get_natural_split_dataset(dataset_name)
    elif split_name == "labelnoise" or split_name == "featurenoise":
        client_sets, testset, metadata = get_simulated_split_dataset(
            dataset_name, 'iid', num_clients=num_clients, split_config=split_config
        )
    else:
        client_sets, testset, metadata = get_simulated_split_dataset(
            dataset_name, split_name, num_clients=num_clients, split_config=split_config
        )
    os.environ["NUM_CLASSES"] = str(metadata["num_classes"])
    os.environ["IN_CHANNELS"] = str(metadata["in_channels"])

    # FIXME:

    if split_name == "labelnoise":
        client_sets = add_label_noise_to_datasets(client_sets, split_config['noise_flip_percent'])
    elif split_name == "featurenoise":
        client_sets = add_feature_noise_to_datasets(client_sets, split_config['noise_mu'], split_config['noise_sigma'] )
    
    model_base = get_clf_model(model_name, metadata)

    return client_sets, testset, model_base


def get_full_dataset_and_model(dataset_name: str, model_name: str):


    train, test, metadata = get_standard_clf_datasets(dataset_name)
    os.environ["NUM_CLASSES"] = str(metadata["num_classes"])
    os.environ["IN_CHANNELS"] = str(metadata["in_channels"])

    model_base = get_clf_model(model_name, metadata)

    return train, test, model_base
