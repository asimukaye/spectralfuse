from pathlib import Path
import os  
import yaml
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import argparse
import time
import torch
import numpy as np
from itertools import combinations
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from pandas import json_normalize
import weightwatcher as ww
from scipy.stats import kendalltau
# For CIFAR-10
from torchvision.models import resnet18, resnet50, ResNet
from rewards import interpolation_rewards, sparsification_gradient_rewards, sparsification_param_rewards, no_rewards

from data import get_fl_datasets_and_model

from trainutils import train_one_epoch_model, evaluate_model, adapt_model_last_layer
from typing import Literal, Optional

from utils import (
    append_to_ledger,
    setup_output_dirs,
    set_seed,
    auto_configure_device,
    json_dump,
    yaml_dump,
    FLLogger
)

from configdefs import Config


def recalibrate_bn(
    model: nn.Module,
    dataloader: DataLoader,
    batch_size,
    num_batches=10,
    device=torch.device("cuda"),
):
    model.to(device)
    model.train()
    dataloader = DataLoader(
        dataloader.dataset, batch_size=batch_size, shuffle=True
    )  # Use the original dataset for recalibration
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(x.to(device))  # forward pass only to update BN buffers
    # model.eval()  # Set the model back to evaluation mode
    return model


@dataclass
class SpectralFuseLogger(FLLogger):
    client_ww_details: dict[int, list] = field(default_factory=dict)
    client_ww_summary: dict[int, list] = field(default_factory=dict)
    server_ww_details: list = field(default_factory=list)
    server_ww_summary: list = field(default_factory=list)
    ww_times : list = field(default_factory=list)
    ww_perf_times : list = field(default_factory=list)

    client_entropy_ll_sclr: dict[int, list] = field(default_factory=dict)
    client_weights_sclr: dict[int, list] = field(default_factory=dict)
    # client_weights_lyr: dict[int, list] = field(default_factory=dict)
    client_shapleys_sclr: dict[int, list] = field(default_factory=dict)
    client_raw_shapleys_sclr: dict[int, list] = field(default_factory=dict)
    client_entropy_smooth_sclr: dict[int, list] = field(default_factory=dict)



    def __post_init__(self):
        # Initialize client and server weight watcher details
        self.client_ww_details = {i: [] for i in range(self.num_clients)}
        self.client_ww_summary = {i: [] for i in range(self.num_clients)}
        self.client_entropy_ll_sclr  = {i: [] for i in range(self.num_clients)}
        self.client_weights_sclr = {i: [] for i in range(self.num_clients)}
        # self.client_weights_lyr = {i: [] for i in range(self.num_clients)}
        self.client_shapleys_sclr = {i: [] for i in range(self.num_clients)}
        self.client_raw_shapleys_sclr = {i: [] for i in range(self.num_clients)}
        self.client_entropy_smooth_sclr = {i: [] for i in range(self.num_clients)}
        super().__post_init__()


@dataclass
class SpectralFuseConfig(Config):
    method: str = "linear_weights"  # "linear_weights" or "softmax_weights"
    mu: float = 0.9
    rewards: str = "none"  # "interpolation" or "sparsification" or "none"
    layer_prefix: str = "fc"  # Layer prefix for class-wise Shapley values
    participation_rate: float = 1.0  # fraction of clients participating each round

    
    def __post_init__(self):
        self.expt_obj = f"{self.method}_mu{self.mu}"
        if "vit" in self.model_name:
            self.layer_prefix = "classifier"
        elif "tf_cnn" in self.model_name:
            self.layer_prefix = "fc2"
        elif "mlpnet" in self.model_name:
            self.layer_prefix = "fc3"
        else:
            self.layer_prefix = "fc"

        super().__post_init__()


# ---------- utilities ----------

def rankdata(arr):
    """Return 1-based ranks (handle ties by average rank). Pure numpy implementation."""
    arr = np.asarray(arr)
    order = arr.argsort(kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float) + 1  # 1-based
    # handle ties: average ranks for equal values
    # find unique values and average ranks
    unique_vals, inv_idx = np.unique(arr, return_inverse=True)
    if len(unique_vals) != len(arr):
        # compute average rank per unique value
        avg_ranks = np.zeros(len(unique_vals), dtype=float)
        for u in range(len(unique_vals)):
            avg_ranks[u] = ranks[arr == unique_vals[u]].mean()
        ranks = avg_ranks[inv_idx]
    return ranks

def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0:
        return 0.0
    return float((xm*ym).sum() / denom)

def robust_zscore(x, eps=1e-6):
    """Standardize vector x across clients between 0 to 1."""
    x = np.asarray(x, dtype=float)
    # med = np.median(x)
    # mad = np.median(np.abs(x - med)) + eps
    print(x)
    # return (x - med) / mad
    mu = np.mean(x)
    print(np.std(x))
    s = np.std(x) + eps
    print(s/mu)
    if s/mu < eps:  # avoid division by zero; if very small variance, return uniform weights
        return np.ones_like(x)*(1.0/len(x))
    
    return (x - mu) / s
def sum_to_one(x, eps=1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x/(x.sum() + eps)
# ---------- Kalman fusion class ----------

class RankAdaptiveKalman:
    def __init__(self, n_clients, Q=1e-4, beta=0.1,
                 scale=1.0, min_var=1e-3, max_var=100.0, init_P=1.0):
        self.n = n_clients
        self.Q = Q                # process noise (scalar)
        self.beta = beta          # EMA smoothing for rank-corr
        self.scale = scale        # map (1 - r) -> variance
        self.min_var = min_var
        self.max_var = max_var

        # state per client
        self.x = np.ones(n_clients)*(1/n_clients)  + Q*np.random.randn(n_clients)    # state mean (initial)
        self.P = np.ones(n_clients) * init_P  # state covariance (scalar per client)

        # EMA-smoothed rank correlations for metrics a and b
        self.r_a = 0.0
        self.r_b = 0.0

    def _map_corr_to_var(self, r_ema: np.ndarray):
        # linear mapping; you may adjust to non-linear if desired
        var = self.scale * (1.0 - r_ema) + self.min_var
        return np.clip(var, self.min_var, self.max_var)

    def update(self, a_raw, b_raw, mask=None):
        """
        a_raw, b_raw: arrays of length n (metrics for all clients at current round)
        returns fused x (length n)
        """
        if mask is None:
            mask = np.ones(self.n, dtype=bool)

        # predict
        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        # only use participating clients for measurements
        a_sel = np.asarray(a_raw, dtype=float)[mask]
        b_sel = np.asarray(b_raw, dtype=float)[mask]
        print(f"a_sel: {a_sel.round(4)}, \nb_sel: {b_sel.round(4)}")
        x_sel = x_pred[mask]

        if len(a_sel) == 0:
            # no update possible
            self.x = x_pred
            self.P = P_pred
            return self.x.copy(), {'r_a': self.r_a, 'r_b': self.r_b,
                                   'var_a': self._map_corr_to_var(self.r_a),
                                   'var_b': self._map_corr_to_var(self.r_b)}

        a_std = sum_to_one(a_sel)
        b_std = sum_to_one(b_sel)
        print(f"a_std: {a_std.round(4)}, \nb_std: {b_std.round(4)}")
        # Kendall Tau version (slower)
        # r_a_now = kendalltau(a_std, x_pred).correlation # type: ignore
        # r_b_now = kendalltau(b_std, x_pred).correlation # type: ignore

        ranks_a = rankdata(a_std)
        ranks_b = rankdata(b_std)
        ranks_x = rankdata(x_sel)

        r_a_now = pearson_corr(ranks_a, ranks_x)  # Spearman-like
        r_b_now = pearson_corr(ranks_b, ranks_x)
        # print(f"r_a_now: {r_a_now}, r_b_now: {r_b_now}")

        # if 
        # 3) EMA smoothing of correlations
        self.r_a = (1 - self.beta) * self.r_a + self.beta * r_a_now
        self.r_b = (1 - self.beta) * self.r_b + self.beta * r_b_now

        # 4) map to variances
        var_a = self._map_corr_to_var(self.r_a)
        var_b = self._map_corr_to_var(self.r_b)
        # measurement cov matrix is same for all clients this round (can be made per-client if desired)
        # R = diag(var_a, var_b)

        # Precompute for Kalman gain: H = [1,1]^T
        # For scalar P_pred: S = H P H^T + R = [[P+var_a, P],[P, P+var_b]]
        # We'll invert S explicitly (2x2) and compute K = P * H^T * S^{-1}
        Pp = P_pred[mask]  # vector length n for selected clients
        # For each client we need K (1x2) : K = Pp * [1,1]^T @ S^{-1}
        # but S is same structure for all clients except Pp differs; implement vectorized.
        # Inverse of S = (1/det) * [[P+var_b, -P],[-P, P+var_a]] where det = (P+var_a)(P+var_b) - P^2
        Pa = Pp + var_a
        Pb = Pp + var_b
        det = Pa * Pb - (Pp * Pp)
        # avoid tiny det
        det = np.where(det == 0, 1e-12, det)

        # S^{-1} * (y - H x_pred) can be computed; we'll compute K * residual directly:
        # K = Pp * [ (Pa/det), (-Pp/det) ]  ??? careful: K = Pp * H^T S^{-1}
        # Let's compute S^{-1} = 1/det * [[Pb, -Pp],[-Pp, Pa]]
        # H^T S^{-1} = [1,1] @ S^{-1} = 1/det * [ (Pb - Pp), (-Pp + Pa) ] = 1/det * [Pb - Pp, Pa - Pp]
        # But Pa - Pp = var_a, Pb - Pp = var_b
        # So H^T S^{-1} = 1/det * [var_b, var_a]
        # Therefore K = Pp * (1/det) * [var_b, var_a]   (1x2 vector)
        k_factor = Pp / det  # vector
        K1 = k_factor * var_b  # coefficient for metric a
        K2 = k_factor * var_a  # coefficient for metric b

        # residual: y - H x_pred = [a_std - x_pred, b_std - x_pred] (per client)
        res_a = a_std - x_sel
        res_b = b_std - x_sel

        # update state
        x_upd_sel = x_sel + K1 * res_a + K2 * res_b

        # update covariance: P = Pp - K * H * Pp = Pp - (K1 + K2) * Pp
        P_upd_sel = Pp - (K1 + K2) * Pp
        # numerical safeguards
        P_upd_sel = np.clip(P_upd_sel, 1e-8, 1e6)

        # save back
        self.x = x_pred.copy()
        self.P = P_pred.copy()
        self.x[mask] = x_upd_sel
        self.P[mask] = P_upd_sel

        return self.x.copy(), {'r_a': self.r_a, 'r_b': self.r_b,
                               'var_a': var_a, 'var_b': var_b}

EPSILON = 1e-10 # only in normal precision

def matrix_rank(svals, N, tol=None):
    """Matrix rank, computed from the singular values directly

    svals are the singular values
    N is the largest dimension of the matrix
 
    re-implements np.linalg.matrix_rank(W) """
    S = svals
    if tol is None:
        tol = np.max(S) * N * np.finfo(S.dtype).eps
    return np.count_nonzero(S > tol)


def matrix_entropy(svals, N):
    """Matrix entropy of real, computed using the singular values, and the dim N"""

    entropy = -1
    
    try:
        svals = np.sqrt(svals) 
        rank = matrix_rank(svals, N) #np.linalg.matrix_rank(W)
    
        evals = svals*svals
        p = evals / np.sum(evals) + EPSILON
        rank += EPSILON
        entropy = -np.sum(p * np.log(p)) / np.log(rank) 
        
    except (ZeroDivisionError, ValueError) as e:
        # Handle divide by zero and invalid value errors
        # logger.warning("Error:", e)
        print("Error:", e)
    except Exception as e:
        # Handle other fatal errors
        # logger.warning("Error:", e)
        print("Error:", e)
    
    return entropy

def compute_cssv_cifar(client_deltas: dict[int, nn.Module], weights, num_classes, layer_prefix, selected_clients):
    # all_client_grads = [c.state_dict() for c in client_deltas.values()]
    weights = np.array([weights[i] for i in selected_clients])
    all_client_grads = [{name: param for name, param in c.named_parameters()} for cid, c in client_deltas.items() if cid in selected_clients]
    n = len(all_client_grads)
    # num_classes = 10 # clients[0].model.state_dict()['linear.weight'].shape[0]
    similarity_matrix = torch.zeros((n, num_classes))  # One similarity value per class

    weight_layer_name = f"{layer_prefix}.weight"
    bias_layer_name = f"{layer_prefix}.bias"
    subsets = [subset for subset in combinations(range(n), n)]


    for subset in subsets:
        # Create a temporary server for this subset
        subset_grads = [all_client_grads[i] for i in subset]
        # subset_clients = [clients[i] for i in subset]
        curr_weights = [weights[j] for j in subset]
        # normalized_curr_weights = softmax(curr_weights)  # curr_weights / np.sum(curr_weights)
        normalized_curr_weights = curr_weights / np.sum(curr_weights)

        # print(f"Normalized current weights: { normalized_curr_weights}")

        # temp_server = Server(subset_clients, model_template)
        total_grads = None

        for client_id, client_grads in enumerate(all_client_grads):

            if total_grads is None:
                total_grads = {
                    name: torch.zeros_like(grad) for name, grad in client_grads.items()
                }

            for name, grad in client_grads.items():
                total_grads[name] += weights[client_id] * grad

        for cls_id in range(num_classes):
            

            w1_grad = torch.cat(
                [
                    total_grads[weight_layer_name][cls_id].view(-1),
                    total_grads[bias_layer_name][cls_id].view(-1),
                ]
            ).view(1, -1)

            w1_grad = F.normalize(w1_grad, p=2)

            # print(f"Class {cls_id}, Gradient Norm: {torch.norm(w1_grad).item():.4f}"    )

            for client_id in range(len(subset)):
                w2_grad = torch.cat(
                    [
                        subset_grads[client_id][weight_layer_name][cls_id].view(-1),
                        subset_grads[client_id][bias_layer_name][cls_id].view(-1),
                    ]
                ).view(1, -1)
                w2_grad = F.normalize(w2_grad, p=2)

                # print(f"Client {client_id}, Class {cls_id}, Gradient Norm: {torch.norm(w2_grad).item():.4f}")

                # Compute cosine similarity with gradients
                sim = F.cosine_similarity(w1_grad, w2_grad).item()
                sim = (1.0 + sim) / 2  # Normalize to [0, 1]
                similarity_matrix[client_id][cls_id] = sim
                # print(f"Client {client_id}, Class {cls_id}, Similarity: {sim:.4f}" )

    shapley_values = torch.mean(similarity_matrix, dim=1).numpy()
    return shapley_values, similarity_matrix


def weight_to_matrix(W: torch.Tensor) -> torch.Tensor:
    """
    Convert a layer weight tensor to a 2D matrix M.
    - Linear: (out, in) -> (out, in)
    - Conv: (out_ch, in_ch, kH, kW, ...) -> (out_ch, in_ch * kH * kW * ...)
    - Anything else: first dim as rows, flatten the rest.
    """
    if W.ndim == 2:
        return W
    if W.ndim >= 2:
        return W.reshape(W.shape[0], -1)
    # 1D (e.g., bias) -> treat as (n,1)
    return W.reshape(-1, 1)


def gram_psd(M: torch.Tensor, side: Literal["auto", "left", "right"] = "auto") -> torch.Tensor:
    """
    Build a symmetric PSD Gram matrix:
      left:  A = M M^T  (size: rows x rows)
      right: A = M^T M  (size: cols x cols)
      auto: choose smaller of the two for cheaper eigendecomp.
    """
    r, c = M.shape
    if side == "auto":
        side = "left" if r <= c else "right"

    if side == "left":
        A = M @ M.transpose(-1, -2)
    else:
        A = M.transpose(-1, -2) @ M
    # Force symmetry (numerical noise)
    return 0.5 * (A + A.transpose(-1, -2))

def von_neumann_entropy_from_psd(
    A: torch.Tensor,
    normalize: Literal["none", "fro", "trace"] = "fro",
    eps: float = 1e-12,
    log_base: float = torch.e,  # use 2.0 for bits
) -> torch.Tensor:
    """
    Compute S(A) = -sum_j λ_j log λ_j from a symmetric PSD matrix A.
    normalize:
      - "none": no normalization
      - "fro":  A <- A / ||A||_F          (common in practice)
      - "trace": A <- A / trace(A)        (density-matrix style; λ sum to 1)
    """
    if normalize == "fro":
        denom = torch.linalg.norm(A, ord="fro").clamp_min(eps)
        A = A / denom
    elif normalize == "trace":
        tr = torch.trace(A).clamp_min(eps)
        A = A / tr

    # Eigenvalues of symmetric matrix
    evals = torch.linalg.eigvalsh(A)

    # Clamp to avoid log(0) and tiny negative numerical artifacts
    evals = evals.clamp_min(eps)

    # Change of base: log_b(x) = ln(x)/ln(b)
    ln_base = torch.log(torch.as_tensor(log_base, device=evals.device, dtype=evals.dtype))
    ent = -(evals * (torch.log(evals) / ln_base)).sum()
    return ent


def layer_spectral_entropy(
    W: torch.Tensor,
    gram_side: Literal["auto", "left", "right"] = "auto",
    normalize: Literal["none", "fro", "trace"] = "fro",
    eps: float = 1e-12,
    log_base: float = torch.e,
) -> torch.Tensor:
    """
    End-to-end spectral entropy for a layer weight tensor W (PyTorch).
    Returns a scalar tensor.
    """
    M = weight_to_matrix(W).to(dtype=torch.float64)  # improve stability
    A = gram_psd(M, side=gram_side)
    return von_neumann_entropy_from_psd(A, normalize=normalize, eps=eps, log_base=log_base).to(W.dtype)


def run_spectralfuse(cfg: SpectralFuseConfig):
    start_time = time.time()

    logger = SpectralFuseLogger(
            num_clients=cfg.num_clients, output_dir=cfg.output_dir, use_tensorboard=False
        )
   
    if cfg.resumed:
        logger.resume_logger(cfg.output_dir)
        print("Resumed logger from existing directory.")
        start_round = logger.curr_round[-1] + 1
        print(f"Current round: {logger.curr_round[-1]+1}")

        # input("Press Enter to continue...")
    else:
        start_round = 0
    

    client_sets, test_dataset, model_base = get_fl_datasets_and_model(
        dataset_name=cfg.dataset_name,
        model_name=cfg.model_name,
        split_name=cfg.split,
        num_clients=cfg.num_clients,
        split_config=cfg.split_cfg
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    num_classes = int(os.environ["NUM_CLASSES"])
    client_models: dict[int, nn.Module] = {}
    client_deltas: dict[int, nn.Module] = {}
    # client_deltas: dict[int, nn.Module] = {}
    client_train_loaders = {}
    client_val_loaders = {}
    # if 'fedopt' in cfg.strategy:
    # agg_gradient = [torch.zeros_like(param.data) for param in model_base.parameters()]
    kalman_filter = RankAdaptiveKalman(cfg.num_clients, Q=1e-4, beta=0.1, scale=1.0, min_var=1e-3, max_var=100.0, init_P=1.0)
    num_selected_clients = max(1, int(cfg.participation_rate * cfg.num_clients))
    for i in range(cfg.num_clients):
        client_models[i] = deepcopy(model_base)
        if cfg.resumed:
            client_models[i].load_state_dict(
                torch.load(cfg.output_dir / f"model_{i}.pth")
            )
        client_deltas[i] = deepcopy(model_base)
        client_deltas[i].to(cfg.device)
        # train_indices = client_train_val_indices[i][0]
        # train_indices = client_train_val_indices[i]
        # client_train_loaders[i] = DataLoader(
        #     Subset(dataset, train_indices), batch_size=batch_size, shuffle=True  # type: ignore
        # )
        client_train_loaders[i] = DataLoader(client_sets[i], batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        # val_indices = client_train_val_indices[i][1]
        # client_val_loaders[i] = DataLoader(
        #     Subset(dataset, val_indices), batch_size=batch_size, shuffle=False  # type: ignore
        # )

    global_model = deepcopy(model_base)
    if cfg.resumed:
        global_model.load_state_dict(torch.load(cfg.output_dir / "global_model.pth"))
    global_model.to(cfg.device)
    

    ## ADD RESUME LOGIC HERE
    if cfg.resumed:
        agg_weights = np.array(list(logger.client_weights_sclr.values()))[:,-1]  # Load the last saved aggregation weights
    else:
        agg_weights = 1/cfg.num_clients*np.ones(cfg.num_clients)  # Initialize aggregation weights
    print(f"Aggregation Weights Shape: {agg_weights.shape}")


    
    # fedavg_weights = np.array(data_lengths) / np.sum(data_lengths)

    if cfg.resumed:
        start_lr = logger.lr_summary[-1]
        print(f"Resumed learning rate: {start_lr}")
    else:
        start_lr = cfg.lr

    server_optimizer = cfg.optimizer_partial(global_model.parameters(), lr=cfg.lr)

    scheduler = (
        cfg.scheduler_partial(server_optimizer)
        if isinstance(cfg.scheduler, type)
        else None
    )
    # Logging 

    json_dump(server_optimizer.__dict__['defaults'], cfg.output_dir / "optimizer_cfg.json")
    if scheduler is not None:
        scheduler_attributes = scheduler.__dict__.copy()
        scheduler_attributes.pop('optimizer', None)  # Remove the optimizer from the dict
        json_dump(scheduler_attributes, cfg.output_dir / "scheduler_cfg.json")
        torch.save(scheduler.state_dict(), cfg.output_dir / "scheduler.pth")

    cfg.save_config()
    total_time = []
    entropies = np.full(cfg.num_clients, np.nan, dtype=float)
    temp_shapley_values = np.full(cfg.num_clients, np.nan, dtype=float)

    shapley_values = np.full(cfg.num_clients, np.nan, dtype=float)
    class_shapley_values = np.full(cfg.num_clients, np.nan, dtype=float)  # to store class-wise Shapley values
    smooth_entropies = np.full(cfg.num_clients, np.nan, dtype=float)
    for r in range(start_round, cfg.num_rounds):
        print(f"Round {r + 1}/{cfg.num_rounds}")
        logger.curr_round.append(r)

        client_params = {}
        
        # shapleys = np.empty_like(agg_weights)
        if scheduler is not None:
            lr = scheduler.get_last_lr()[0]
            logger.lr_summary.append(lr)
        else:
            lr = cfg.lr

        selected_clients = np.random.choice(
            range(cfg.num_clients), 
            size=num_selected_clients, 
            replace=False
        )
        selected_mask = np.zeros(cfg.num_clients, dtype=bool)
        selected_mask[selected_clients] = True

        for i, client in client_models.items():
            if i not in selected_clients:
                continue

            print(f"Training client {i}/{cfg.num_clients}")
            client.to(cfg.device)
            client.train()  # Set the client model to training mode


            optimizer = cfg.optimizer_partial(
                client.parameters(),
                lr=lr,
            )
            for e in range(cfg.num_epochs):
                client_model_copy = deepcopy(client.state_dict())
                client, losses, acc, bacc = train_one_epoch_model(client, client_train_loaders[i], optimizer, cfg.criterion, cfg.device)
                if np.isnan(losses).sum() > 0:
                    print(f"NaN values found in training losses for client {i} at epoch {e}!")
                    # restore model
                    client.zero_grad()
                    client.load_state_dict(client_model_copy)
                logger.client_trn_losses[i].extend(losses)
                logger.client_trn_accs[i].append(acc)
                logger.client_trn_baccs[i].append(bacc)

            test_acc, test_bacc, f1_wtd, f1_micro,  test_losses = evaluate_model(
                client, test_loader, cfg.criterion, cfg.device
            )
            logger.client_test_accs[i].append(test_acc)
            logger.client_test_baccs[i].append(test_bacc)
            logger.client_test_f1wtd[i].append(f1_wtd)
            logger.client_test_f1mic[i].append(f1_micro)
            logger.client_test_losses[i].extend(test_losses)

            # Find the difference between the client model and the global model
            delta_model = client.state_dict()
            # grad_model = client.state_dict()
            for k, v in global_model.named_parameters():
                delta_model[k] = delta_model[k] - v.data
                # grad_model[k] = -1/lr *delta_model[k]  # Approximate gradient

            client_deltas[i].load_state_dict(delta_model)
            # client_deltas[i].to("cpu")

            # cww = ww.WeightWatcher(client)
            tick = time.time()
            proc_tick = time.process_time()
            cww = ww.WeightWatcher(client_deltas[i], log_level="ERROR")
            analysis = cww.analyze()
            summary = cww.get_summary()
            tock = time.time()
            proc_tock = time.process_time()
            # print(f"WeightWatcher analysis time for client {i}: {tock - tick:.6f} seconds (wall), {proc_tock - proc_tick:.6f} seconds (CPU)")
            # print(f"Client {i} -  WW Spectral Entropy of layer {cfg.layer_prefix}.weight: {analysis['entropy'].to_numpy()[-1]:.6f}")

            logger.ww_times.append(tock - tick)
            logger.ww_perf_times.append(proc_tock - proc_tick)

            logger.client_ww_details[i].append(analysis.to_numpy()) # type: ignore
            logger.client_ww_summary[i].append(list(summary.values()))
            logger.client_entropy_ll_sclr[i].append(analysis['entropy'].to_numpy()[-1]) # type: ignore


            entropies[i] = analysis['entropy'].to_numpy()[-1] # type: ignore
                # logger.client_weights_sclr[i].append(temp_weights[i,:].copy())

            client_params[i] = client.state_dict()

        tick2 = time.time()
        # proc_tick2 = time.process_time()
        # for i in range(cfg.num_clients):
        #     pure_entropy = layer_spectral_entropy(client_deltas[i].state_dict()[f"{cfg.layer_prefix}.weight"],).item()
        # tock2 = time.time()
        # proc_tock2 = time.process_time()
        # t1 = tock2 - tick2
        # print(f"Spectral Entropy computation time: {tock2 - tick2:.6f} seconds (wall), {proc_tock2 - proc_tick2:.6f} seconds (CPU)")
        # print(f"Client {i} - Manual Spectral Entropy of layer {cfg.layer_prefix}.weight: {pure_entropy:.6f}")

        tick3 = time.time()

        out_shapley, out_class_shapley = compute_cssv_cifar(client_deltas, agg_weights, num_classes, cfg.layer_prefix, selected_clients)

        temp_shapley_values[selected_clients] = out_shapley
        # print(f"temp_shapley_values: {temp_shapley_values.round(4)}")

        # if np.isnan(temp_shapley_values).sum() > 0:
        #     # print warning
        #     print(f"NaN values found in Shapley values! Replacing with previous values.")
        #     temp_shapley_values = shapley_values
        # if np.isnan(temp_class_shapley_values).sum() > 0:
        #     print(f"NaN values found in Class-wise Shapley values! Replacing with previous values.")
        #     temp_class_shapley_values = class_shapley_values


        # if np.isnan(entropies).sum() > 0:
        #     # print warning
        #     print(f"NaN values found in Entropies values! Replacing with previous values.")
        #     entropies = smooth_entropies

        # for i in range(cfg.num_clients):
        #     if i not in selected_clients:
        #         continue
        #     logger.client_raw_shapleys_sclr[i].append(temp_shapley_values[i].copy())

        if np.isnan(shapley_values).sum() > 0 :
            shapley_values[np.isnan(shapley_values)] = temp_shapley_values[np.isnan(shapley_values)]
            # shapley_values = temp_shapley_values.copy()
            # class_shapley_values = np.array(temp_class_shapley_values)
        else:
        # print(f"selected shapley_values before smoothing: {temp_shapley_values[selected_clients].round(4)}")
            shapley_values[selected_clients] = (
                cfg.mu * shapley_values[selected_clients] + (1 - cfg.mu) * temp_shapley_values[selected_clients]
            )
            # class_shapley_values = cfg.mu * class_shapley_values + (
            #     1 - cfg.mu
            # ) * np.array(temp_class_shapley_values)

        if np.isnan(smooth_entropies).sum() > 0 :
            smooth_entropies[np.isnan(smooth_entropies)] = entropies[np.isnan(smooth_entropies)]
        else:
            print(f"Smooth Entropies before: {smooth_entropies.round(4)}")
            smooth_entropies[selected_clients] = (
                (1 - cfg.mu) * entropies[selected_clients]
                + cfg.mu * smooth_entropies[selected_clients]
            )
        
        # print(f"Entropies: {entropies.round(4)}")
        # print(f"Smooth Entropies: {smooth_entropies.round(4)}")
        # print(f"Shapley Values: {shapley_values.round(4)}")


        agg_weights, r_dict = kalman_filter.update(smooth_entropies, shapley_values, mask=selected_mask)
        print(f"Agg_weights _pre: {agg_weights.round(4)}")

        # Normalize
        agg_weights_round = np.zeros_like(agg_weights)
        agg_weights_round[selected_clients] = agg_weights[selected_clients] / (
            agg_weights[selected_clients].sum(axis=0, keepdims=True) + 1e-12
        )
        print(f"Agg_weights final: {agg_weights_round[selected_clients].round(4)}")

        for i in range(cfg.num_clients):
            # if i not in selected_clients:
            #     continue
            logger.client_weights_sclr[i].append(agg_weights[i].copy())
            logger.client_shapleys_sclr[i].append(shapley_values[i].copy())
            logger.client_entropy_smooth_sclr[i].append(smooth_entropies[i].copy())


        # Entropy averaging
        
        for l, (key, param) in enumerate(global_model.named_parameters()):
            temp_parameter = torch.zeros_like(param.data)
            for cid, c_state in client_params.items():
                if cid not in selected_clients:
                    continue
                temp_parameter.data.add_(agg_weights_round[cid] * c_state[key].data)
            param.data.copy_(temp_parameter)

        tock3 = time.time()
        t2 = tock3 - tick2
        # print(f"Kalman filter update time: {tock3 - tick3:.6f} seconds (wall)")
        # if r >0 and r < 11:
        #     total_time.append(t2)
        #     avg_time, std_time = np.mean(total_time), np.std(total_time)
        #     print(f"Average time for WW + SE + KF over rounds 1-{r}: {avg_time:.6f} +- {std_time:.6f} seconds (wall)")

        if scheduler is not None:
            scheduler.step()
        # global_model.load_state_dict(global_state_dict)        # use recalibrate_bn if resnet models
        if isinstance(global_model, ResNet):
            global_model = recalibrate_bn(
                global_model, test_loader, cfg.batch_size, num_batches=5, device=cfg.device
            )
        # global_model = recalibrate_bn(global_model, test_loader, num_batches=5)
        global_acc, global_bacc, global_f1_wtd, global_f1_micro, global_test_loss = evaluate_model(global_model, test_loader, cfg.criterion, cfg.device)

        logger.server_accs.append(global_acc)
        logger.server_baccs.append(global_bacc)
        logger.server_f1wtd.append(global_f1_wtd)
        logger.server_f1mic.append(global_f1_micro)
        logger.server_test_losses.extend(global_test_loss)
        print(f"Global Model - Round {r + 1} - Test Accuracy: {global_acc}")
        print("\n")

        # gww = ww.WeightWatcher(global_model, log_level="ERROR")
        # gdetails = gww.analyze()
        # summary = gww.get_summary()
        # logger.server_ww_details.append(gdetails.to_numpy()) # type: ignore
        # logger.server_ww_summary.append(list(summary.values()))

        
        ### send the global model to all clients
        for cid, client in client_models.items():
            # client_state_dict = client.state_dict()
            if cfg.rewards == "interpolation":
                agg_weights_cid = torch.tensor(agg_weights[cid], device=cfg.device, dtype=torch.float32)
                interpolation_rewards(global_model.parameters(), client.parameters(), coeff=agg_weights_cid)
            elif cfg.rewards == "sparsification":
                agg_weights_cid = torch.tensor(agg_weights[cid], device=cfg.device, dtype=torch.float32)
                sparsification_param_rewards(client.parameters(), global_model.parameters(), coeff=agg_weights_cid, beta=1.0)
            elif cfg.rewards == "none":
                no_rewards(global_model.parameters(), client.parameters())
            else:
                raise ValueError(f"Unknown rewards method: {cfg.rewards}")
            
            # for key, param in client.named_parameters():
            #     # for key, param in client.state_dict().items():
            #     # param.data = global_state_dict[key].data.copy_()
            #     param.data.copy_(global_state_dict[key].data)

        logger.write_to_tensorboard()

        if r % 10 == 0 or r == cfg.num_rounds - 1:

            for i in range(cfg.num_clients):
                # Save the model
                torch.save(
                    client_models[i].state_dict(), cfg.output_dir / f"model_{i}.pth"
                )
            
            # Save the global model
            torch.save(global_model.state_dict(), cfg.output_dir / "global_model.pth")
            
            logger.flush()
            # run_summary = logger.generate_summary()
            # elapsed_time = time.time() - start_time
            # run_summary["elapsed_time"] = np.round(elapsed_time, 2)

            # # Save the run summary
            # json_dump(run_summary, cfg.output_dir / "run_summary.json")

            # all_summary = cfg.main_summary.copy()
            # all_summary.update(run_summary)  # type: ignore

            # append_to_ledger(all_summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--strategy", type=str, default="spectralfuse")
    parser.add_argument( '-d',"--debug", action="store_true")
    parser.add_argument("--split", type=str, default="iid")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="tf_cnn")
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--reward", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="logfix")
    parser.add_argument("--resume_from", type=str, default="")  # Path to checkpoint to resume from

    args = parser.parse_args()

    if args.resume_from != "":
        resumed_run = True
    else:
        resumed_run = False
    # for free_rider_idx in [0,1,2,3,4]:
    #     print("=========================================")
    #     print(f"Running for free_rider_idx: {free_rider_idx}")

    free_rider_idx = 1  # You can change this value as needed
    cfg = SpectralFuseConfig(
        seed=args.seed,
        # device=torch.device(auto_configure_device()),
        strategy=args.strategy,
        # strategy="spectralfed",
        optimizer=optim.SGD,
        criterion=nn.CrossEntropyLoss(),
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        freeze_cfg=False,
        num_rounds=200,
        num_clients=10,
        participation_rate=0.5,
        lr=0.1,
        batch_size=64,
        # dataset_name="fedisic",
        dataset_name=args.dataset,

        # model_name="vit_b_16_pret",
        # model_name="tf_cnn",
        model_name=args.model,
        # model_name="resnet18",
        # split="dirichlet",
        # split_cfg={"alpha": 1.0},
        split_cfg={"min_labels": 1}, 
        # split = 'iid',
        # split_cfg={"free_rider_idx": free_rider_idx, "free_rider_actual_size": 10},
        split=args.split,
        # split='free_rider',
        # method="linear_weights_lwise",
        method=f"linear_weights_{args.suffix}",
        # method=f"free_rider_{free_rider_idx}",
        # rewards="none",
        rewards=args.reward,
        # rewards="interpolation",
        # rewards="sparsification",
        # method="softmax_weights",
        # expt_obj=f"baseline_{args.suffix}",
        resumed=resumed_run,
        # mu=0.9,
        mu=args.mu,
        # notes=f"linear wts free rider {free_rider_idx}",
        notes=f"Timing runs",

    )
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"Resuming from config: {resume_path}")
            cfg.load_config(resume_path/'config.yaml')
        else:
            print(f"No directory found at: {resume_path}")

    
    if args.debug:
        cfg.num_rounds = 12
        cfg.scheduler = None
    
    cfg.print_summary()

    # input("Press Enter to continue...")

    run_spectralfuse(cfg)



# 0.7987 - step label skew
# 0.6081 - dirichlet 0.01
