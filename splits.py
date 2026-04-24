import logging
from typing import Sequence, Protocol
import random
import numpy as np
import torch
from torch.utils.data import Subset, Dataset, ConcatDataset, IterableDataset
from noise import NoisySubset, LabelNoiseSubset, BlackWhiteSubset, add_gaussian_noise


def _lookup_label_indices(index_label_mapping: dict, label: int | str) -> list:
    """Return label indices handling both int and string keys."""
    candidate_keys: list[int | str] = []
    for key in (label, str(label)):
        if key not in candidate_keys:
            candidate_keys.append(key)
    if isinstance(label, str):
        try:
            int_key = int(label)
        except ValueError:
            pass
        else:
            if int_key not in candidate_keys:
                candidate_keys.append(int_key)

    for key in candidate_keys:
        if key in index_label_mapping:
            return index_label_mapping[key]
    raise KeyError(f"Label {label} not found in index_label_mapping")

def get_dirichlet_split(index_label_mapping, num_clients, alpha, num_classes):
    # Taken from FedFisher : https://github.com/Divyansh03/FedFisher/blob/main/data.py#L8

    print('Dirichlet alpha:', alpha)
    # min_size = 0
    # N = len(y)
    net_dataidx_map = {}
    p_client = np.zeros((num_clients, num_classes))
    # np.random.seed(42)

    for i in range(num_clients):
        p_client[i] = np.random.dirichlet(np.repeat(alpha, num_classes))
    client_data_indices = [[] for _ in range(num_clients)]
    # client_data_indices_map = {i: [] for i in range(num_clients)}

    for k in range(num_classes):
        # idx_k = np.where(y == k)[0]
        idx_k = np.asarray(_lookup_label_indices(index_label_mapping, k), dtype=int)
        np.random.shuffle(idx_k)
        proportions = p_client[:, k]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_data_indices = [
            idx_j + idx.tolist()
            for idx_j, idx in zip(client_data_indices, np.split(idx_k, proportions))
        ]

    for j in range(num_clients):
        np.random.shuffle(client_data_indices[j])
        net_dataidx_map[j] = client_data_indices[j]

    total_data_points = sum([len(lv) for lv in index_label_mapping.values()])

    label_map = -1* np.ones(total_data_points, dtype=int)
    for k, v in index_label_mapping.items():
        for idx in v:
            label_map[idx] = int(k)
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(label_map[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    local_sizes = []
    for i in range(num_clients):
        local_sizes.append(len(net_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes / np.sum(local_sizes)

    print("Data statistics: %s" % str(net_cls_counts))
    print("Data ratio: %s" % str(weights))

    return client_data_indices

def get_free_rider_split(index_label_mapping, num_clients, num_classes, free_rider_idx=0, free_rider_actual_size=10, alpha=1.0):

    p_client = np.zeros((num_clients, num_classes))
    # np.random.seed(42)

    for i in range(num_clients):
        p_client[i] = np.random.dirichlet(np.repeat(alpha, num_classes))
    client_data_indices = [[] for _ in range(num_clients)]
    # client_data_indices_map = {i: [] for i in range(num_clients)}

    for k in range(num_classes):
        # idx_k = np.where(y == k)[0]
        idx_k = np.asarray(_lookup_label_indices(index_label_mapping, k), dtype=int)
        np.random.shuffle(idx_k)
        proportions = p_client[:, k]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_data_indices = [
            idx_j + idx.tolist()
            for idx_j, idx in zip(client_data_indices, np.split(idx_k, proportions))
        ]

    # Assign the client with least data as free rider
    # free_riders = np.argsort([len(client_data_indices[x]) for x in range(num_clients)])[:num_free_riders]
    # Client
    free_rider_data = np.random.choice(client_data_indices[free_rider_idx], size=free_rider_actual_size, replace=True)

    # make the free rider have repeated data of size equal to average data size of other clients
    avg_data_size = int(np.mean([len(client_data_indices[x]) for x in range(num_clients) if x != free_rider_idx]))
    repeated_data = []
    while len(repeated_data) < avg_data_size:
        repeated_data.extend(free_rider_data)
    client_data_indices[free_rider_idx] = repeated_data[:avg_data_size]
    return client_data_indices


def get_iid_split_v1(
    dataset: Subset, num_splits: int, seed: int = 42
) -> dict[int, np.ndarray]:
    shuffled_indices = np.random.permutation(len(dataset))

    # get adjusted indices
    split_indices = np.array_split(shuffled_indices, num_splits)

    # construct a hashmap
    split_map = {k: split_indices[k] for k in range(num_splits)}
    return split_map

def get_iid_split_v2(num_clients: int, dataset: ConcatDataset):
    num_samples = len(dataset) // num_clients
    client_data_indices = [
        list(range(i * num_samples, (i + 1) * num_samples)) for i in range(num_clients)
    ]
    return client_data_indices


def get_iid_split_v3(
    num_clients: int,  index_label_mapping: dict) -> list[np.ndarray]:
    data_indices = [v for v in index_label_mapping.values()]

    data_indices = np.concatenate(data_indices)
    shuffled_indices = np.random.permutation(data_indices)
    # get adjusted indices
    split_indices = np.array_split(shuffled_indices, num_clients)
    return split_indices

# All labels uniform increasing data points
def get_step_quantity_split(
    num_clients: int, num_classes: int, index_label_mapping: dict, gamma=0.05
):
    assert(abs(gamma) < 2/(num_clients*(num_clients-1))), "Absolute Gamma must be less than to 2/(num_clients*(num_clients-1))"
    available_indices = {int(label): set(val) for label, val in index_label_mapping.items()}
    client_data_indices_map = {i: [] for i in range(num_clients)}

    base_ratio = (1 / num_clients) - ((num_clients - 1) / 2) * gamma
    # Compute sequence
    label_ratios = np.array([base_ratio + i * gamma for i in range(num_clients)])


    for label in range(num_classes):
        label_indices = available_indices[label]
        label_lens = np.int32(np.floor(label_ratios*len(label_indices)))  # This is the total number of indices for this label

        for cid in range(num_clients):
            chosen_indices = np.random.choice(list(label_indices), label_lens[cid], replace=False) # type: ignore
            label_indices = label_indices.difference(chosen_indices)  # Remove chosen indices from available indices
            client_data_indices_map[cid].extend(chosen_indices)

    client_data_indices = [np.array(indices) for indices in client_data_indices_map.values()]
    for indices in client_data_indices:
        np.random.shuffle(indices)

    return client_data_indices

# Step split
def get_step_label_skew_split(
    num_clients: int,
    num_classes: int,
    dataset,
    min_labels=2,
    min_samples_per_label=1500,
):
    assigned_indices = set()
    client_data_indices = []

    for client_idx in range(num_clients):
        """Assigns each client an increasing number of unique labels."""
        num_labels = min_labels + (client_idx * (num_classes // num_clients))
        labels = np.random.choice(range(num_classes), num_labels, replace=False)

        available_indices = [
            i
            for i, (img, label) in enumerate(dataset)
            if label in labels and i not in assigned_indices
        ]
        client_indices = np.random.choice(
            available_indices,
            min(len(available_indices), num_labels * min_samples_per_label),
            replace=False,
        )  # Limit per client: 1500 samples per label

        print(len(client_indices))

        assigned_indices.update(client_indices)
        client_data_indices.append(client_indices)

    return client_data_indices


def get_step_label_skew_split_v2(
    num_clients: int, num_classes: int, index_label_mapping: dict, min_labels=10
):
    client_data_indices_map = {i: [] for i in range(num_clients)}
    num_labels = {
        i: min_labels + (i * ((num_classes - min_labels) // (num_clients - 1)))
        for i in range(num_clients)
    }

    print("Number of labels per client:", num_labels)
    labels_per_client = {}
    for client_idx in range(num_clients):
        labels = np.random.choice(
            range(num_classes), num_labels[client_idx], replace=False
        )
        labels_per_client[client_idx] = set(labels)

    # Second pass: split each label's indices only among clients that selected it.
    for label in range(num_classes):
        clients_with_label = [
            client_idx
            for client_idx, labels in labels_per_client.items()
            if label in labels
        ]
        if not clients_with_label:
            continue
        label_indices = list(index_label_mapping[str(label)])
        np.random.shuffle(label_indices)
        label_splits = np.array_split(label_indices, len(clients_with_label))
        for split_idx, client_idx in enumerate(clients_with_label):
            client_data_indices_map[client_idx].extend(label_splits[split_idx])

    client_data_indices = [
        np.array(indices) for indices in client_data_indices_map.values()
    ]
    for indices in client_data_indices:
        print(len(indices))
        np.random.shuffle(indices)

    return client_data_indices

# label skew split
def get_label_skew_only_split(
    num_clients: int,
    num_classes: int,
    dataset,
    min_labels=2,
    num_samples_per_label=1500,
):
    assigned_indices = set()
    client_data_indices = []

    for client_idx in range(num_clients):
        """Assigns each client an increasing number of unique labels."""
        num_labels = min_labels + (client_idx * (num_classes // num_clients))
        labels = np.random.choice(range(num_classes), num_labels, replace=False)

        available_indices = [
            i
            for i, (img, label) in enumerate(dataset)
            if label in labels and i not in assigned_indices
        ]
        client_indices = np.random.choice(
            available_indices,
            min(len(available_indices), min_labels * num_samples_per_label),
            replace=False,
        )  # Limit per client: 1500 samples per label

        print(len(client_indices))

        assigned_indices.update(client_indices)
        client_data_indices.append(client_indices)

    return client_data_indices


# label skew split
def get_label_skew_only_split_v2(
    num_clients: int,
    num_classes: int,
    index_label_mapping: dict,
    min_labels=10,
    max_samples_per_label=400,
    min_samples_per_label=80,
):
    available_indices = {
        int(label): set(val) for label, val in index_label_mapping.items()
    }
    client_data_indices = []

    for client_idx in range(num_clients):
        """Assigns each client an increasing number of unique labels."""
        num_labels = min_labels + (
            client_idx * ((num_classes - min_labels) // (num_clients - 1))
        )
        labels = np.random.choice(range(num_classes), num_labels, replace=False)
        # print(labels)
        client_indices_list = []
        client_indices = np.array(client_indices_list)

        for i, label in enumerate(labels):
            # Choose indices for each label
            label_indices = available_indices[label]
            if len(label_indices) < min_samples_per_label:
                continue
            else:
                samples_needed = min_samples_per_label + (num_clients - client_idx) * (
                    (max_samples_per_label - min_samples_per_label) // num_clients
                )

                chosen_indices = np.random.choice(
                    list(label_indices),
                    min(samples_needed, len(label_indices)),
                    replace=False,
                )
            client_indices_list.append(chosen_indices)
            available_indices[label] = available_indices[label].difference(
                chosen_indices
            )
            client_indices = np.concatenate(client_indices_list)
            if len(client_indices) > min_labels * max_samples_per_label:
                print(f"Num labels for client {client_idx} are {i}")
                break
        np.random.shuffle(client_indices)

        
        # print(len(client_indices))
        client_data_indices.append(client_indices)

    min_points = min([len(indices) for indices in client_data_indices])
    print("Minimum data points across clients:", min_points)
    for client_idx in range(num_clients):
        client_data_indices[client_idx] = np.random.choice(client_data_indices[client_idx], min_points, replace=False)
        # print(len(client_data_indices[client_idx]))
    return client_data_indices



def add_feature_noise_to_datasets_bugged(
    client_datasets: list,
    noise_mu: float,
    noise_sigma: float,
    pct_noisy: list,
) -> list:
    num_clients = len(client_datasets)

    if isinstance(noise_mu, list):
        assert (
            len(noise_mu) == num_clients
        ), "Number of noise means should match number of patho clients"
        noise_mu_list = noise_mu
    else:
        noise_mu_list = [noise_mu for _ in range(num_clients)]

    if isinstance(noise_sigma, list):
        assert (
            len(noise_sigma) == num_clients
        ), "Number of noise sigmas should match number of patho clients"
        noise_sigma_list = noise_sigma
    else:
        noise_sigma_list = [noise_sigma for _ in range(num_clients)]


    for idx in range(num_clients):
        client_set = client_datasets[idx]
        # total_client_size = len(client_set)
        # local_indices = list(range(total_client_size))
        # selected_indices = np.random.choice(
        #     local_indices, size=int(pct_noisy[idx] * total_client_size), replace=False
        # )
        patho_train = NoisySubset(
            client_set, noise_mu_list[idx], noise_sigma_list[idx], pct_noisy[idx]
        )
        for _ in range(10):
            random_idx = np.random.randint(0, len(client_set))
            img, label = client_set[random_idx]

            new_img, label = patho_train[random_idx]
            print(f"Client {idx}, {torch.norm(new_img-img)}")

        client_datasets[idx] = patho_train
    return client_datasets

def feature_blackout(
    client_datasets: list,
    pct_noisy: list,
) -> list:
    num_clients = len(client_datasets)

    for idx in range(num_clients):
        client_set = client_datasets[idx]
        # total_client_size = len(client_set)
        # local_indices = list(range(total_client_size))
        # selected_indices = np.random.choice(
        #     local_indices, size=int(pct_noisy[idx] * total_client_size), replace=False
        # )
        patho_train = BlackWhiteSubset(
            client_set, pct_noisy[idx]
        )
        for _ in range(5):
            random_idx = np.random.randint(0, len(client_set))
            img, label = client_set[random_idx]

            new_img, label = patho_train[random_idx]
            print(f"Client {idx}, {torch.norm(new_img-img)}")

        client_datasets[idx] = patho_train
    return client_datasets


def add_gaussian_noise_on_root(feat: torch.Tensor, mu: float, sigma: float):
    max_val = torch.max(feat)
    min_val = torch.min(feat)
    feat_range = max_val - min_val
    # Assume feature range is 6 sigma (3 sigma on each side of mean)
    sigma_new = feat_range / 6 * sigma
    noise = torch.randn(feat.size()) * sigma_new + mu
    noisy_feat = feat + noise.to(feat.dtype).to(feat.device)
    noisy_feat = torch.clamp(noisy_feat, min_val, max_val)
    return noisy_feat


def add_feature_noise_to_datasets(
    client_datasets: list[Subset],
    noise_mu: float| list,
    noise_sigma: float| list,
    pct_noisy: list,
) -> list:
    num_clients = len(client_datasets)

    if isinstance(noise_mu, list):
        assert (
            len(noise_mu) == num_clients
        ), "Number of noise means should match number of patho clients"
        noise_mu_list = noise_mu
    else:
        noise_mu_list = [noise_mu for _ in range(num_clients)]

    if isinstance(noise_sigma, list):
        assert (
            len(noise_sigma) == num_clients
        ), "Number of noise sigmas should match number of patho clients"
        noise_sigma_list = noise_sigma
    else:
        noise_sigma_list = [noise_sigma for _ in range(num_clients)]

    for idx in range(num_clients):
        client_set = client_datasets[idx]
        total_client_size = len(client_set)
        local_indices = list(range(total_client_size))
        noisy_indices = np.random.choice(
            local_indices, size=int(pct_noisy[idx] * total_client_size), replace=False
        )
        for i in range(total_client_size):
            data_idx = client_set.indices[i]
            # ds_img, lbl = client_set.dataset[data_idx]
            # dt_lbl = client_set.dataset.targets[data_idx]
            # sb_img, label = client_set[i]

            # print("Labels:", lbl, label, dt_lbl)
            
            
            # print("before:", torch.min(ds_img))
            # print("before:", torch.min(sb_img))
            # print("before:", torch.min(dt_img))

            # print("before:", torch.max(ds_img))
            # print("before:", torch.max(sb_img))
            # print("before:", torch.max(dt_img))

            # print("before:", torch.norm(ds_img-sb_img))
            # print("before:", torch.norm(dt_img-sb_img))
            # print("before:", torch.norm(ds_img-dt_img))
            if i in noisy_indices:
                dt_img = client_set.dataset.data[data_idx] # type: ignore
                new_img = add_gaussian_noise_on_root(dt_img, noise_mu_list[idx], noise_sigma_list[idx])

                client_set.dataset.data[data_idx] = new_img # type: ignore

            # test_img, label = client_set[i]

            # print(torch.norm(new_img-dt_img))
            # # print(torch.norm(test_img-ds_img))
            # # print(torch.norm(test_img-new_img))
            # print(torch.norm(test_img-sb_img))
            # print('\n')

    return client_datasets

def add_label_noise_to_datasets_old(
    client_datasets: list,
    noise_flip_percent: float,
) -> list:
    num_clients = len(client_datasets)
    if isinstance(noise_flip_percent, list):
        assert (
            len(noise_flip_percent) >= num_clients
        ), "Number of noise flip percent should match number of patho clients"
        noise_list = noise_flip_percent
    else:
        noise_list = [noise_flip_percent for _ in range(num_clients)]

    for idx in range(num_clients):
        client_set = client_datasets[idx]
        patho_train = LabelNoiseSubset(client_set, noise_list[idx]) # type: ignore
 
        client_datasets[idx] = patho_train

    return client_datasets


def add_label_noise_to_datasets(
    client_datasets: list[Subset],
    index_label_mapping: dict,
    split_map: list,
    noise_flip_percent: float | list,
    random_flip = True,
) -> list:
    num_clients = len(client_datasets)
    
    if isinstance(noise_flip_percent, list):
        assert (
            len(noise_flip_percent) >= num_clients
        ), "Number of noise flip percent should match number of patho clients"
        noise_list = noise_flip_percent
    else:
        noise_list = [noise_flip_percent for _ in range(num_clients)]

    total_data_points = sum([len(lv) for lv in index_label_mapping.values()])
    label_map = -1* np.ones(total_data_points, dtype=int)

    for k, v in index_label_mapping.items():
        for idx in v:
            label_map[idx] = int(k)

    for idx in range(num_clients):
        client_set = client_datasets[idx]
        total_client_size = len(split_map[idx])
        local_indices = list(range(total_client_size))
        # print(int(noise_list[idx] * total_client_size))
        selected_indices = np.random.choice(
            local_indices, size=int(noise_list[idx] * total_client_size), replace=False
        )
        # print(len(selected_indices), total_client_size, noise_list[idx])
        # split_map[idx]
        assert np.all(client_set.indices == split_map[idx]), "Client dataset indices do not match split map"

        all_labels = [int(k) for k in index_label_mapping.keys()]
        local_indices = list(range(total_client_size))

        for local_idx  in selected_indices:
            dataset_idx = client_set.indices[local_idx]
            # print(f"Client {idx}, Local idx {local_idx}, Dataset idx {dataset_idx}")
            lbl = label_map[dataset_idx]
            # img, lbl = client_set[local_idx]
            # print(f"Before flipping: Client {idx}, Local idx {local_idx}, Dataset idx {dataset_idx}, Label {lbl}, True label {label_map[dataset_idx]}")
            # assert lbl == client_set.dataset.targets[dataset_idx], f"Labels do not match: {lbl} != {client_set.dataset.targets[dataset_idx]}"

            if random_flip:
                excluded_labels = [cid for cid in all_labels if cid != lbl]
                client_set.dataset.targets[dataset_idx] = np.random.choice(excluded_labels) # type: ignore
            else:
                new_label = (lbl + 1) % len(all_labels)
                client_set.dataset.targets[dataset_idx] = new_label # type: ignore
            # new_label = client_set[local_idx][1]
            # print(f"After flipping: Client {idx}, Local idx {local_idx}, Dataset idx {dataset_idx}, New Label {new_label}, True label {label_map[dataset_idx]}")
        # patho_train = LabelNoiseSubset(client_set, noise_list[idx])

        # client_datasets[idx] = patho_train

    return client_datasets
