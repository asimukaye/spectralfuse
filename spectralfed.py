
from pathlib import Path
import os  
import yaml
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import argparse
import time
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from pandas import json_normalize
import weightwatcher as ww

# For CIFAR-10
from torchvision.models import resnet18, resnet50, ResNet
from rewards import interpolation_rewards, sparsification_gradient_rewards, sparsification_param_rewards, no_rewards

from data import get_fl_datasets_and_model

from spectralfuse import layer_spectral_entropy
from trainutils import train_one_epoch_model, evaluate_model, adapt_model_last_layer

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
class SpectralFedLogger(FLLogger):
    client_ww_details: dict[int, list] = field(default_factory=dict)
    client_ww_summary: dict[int, list] = field(default_factory=dict)
    server_ww_details: list = field(default_factory=list)
    server_ww_summary: list = field(default_factory=list)
    client_entropy_ll_sclr: dict[int, list] = field(default_factory=dict)
    client_weights_sclr: dict[int, list] = field(default_factory=dict)
    client_weights_lyr: dict[int, list] = field(default_factory=dict)


    def __post_init__(self):
        # Initialize client and server weight watcher details
        self.client_ww_details = {i: [] for i in range(self.num_clients)}
        self.client_ww_summary = {i: [] for i in range(self.num_clients)}
        self.client_entropy_ll_sclr  = {i: [] for i in range(self.num_clients)}
        self.client_weights_sclr = {i: [] for i in range(self.num_clients)}
        self.client_weights_lyr = {i: [] for i in range(self.num_clients)}
        super().__post_init__()


@dataclass
class SpectralFedConfig(Config):
    method: str = "linear_weights"  # "linear_weights" or "softmax_weights"
    alpha: float = 1.0  # Scaling factor for softmax weights
    mu: float = 0.9
    rewards: str = "none"  # "interpolation" or "sparsification" or "none"
    participation_rate: float = 1.0  # fraction of clients participating each round
    
    def __post_init__(self):
        self.expt_obj = f"{self.method}_{self.alpha}" if "softmax_weights" in self.method else self.method
        if "vit" in self.model_name:
            self.layer_prefix = "classifier"
        elif "tf_cnn" in self.model_name:
            self.layer_prefix = "fc2"
        elif "mlpnet" in self.model_name:
            self.layer_prefix = "fc3"
        else:
            self.layer_prefix = "fc"
        super().__post_init__()


def run_spectralfed(cfg: SpectralFedConfig):
    start_time = time.time()


    logger = SpectralFedLogger(
            num_clients=cfg.num_clients, output_dir=cfg.output_dir, use_tensorboard=False
        )
   
    client_sets, test_dataset, model_base = get_fl_datasets_and_model(
        dataset_name=cfg.dataset_name,
        model_name=cfg.model_name,
        split_name=cfg.split,
        num_clients=cfg.num_clients,
        split_config=cfg.split_cfg
    )

    layernames = [k for k, v in model_base.named_parameters()]
    num_selected_clients = max(1, int(cfg.participation_rate * cfg.num_clients))

    if 'lwise' in cfg.method:
        # print(len([p for p in model_base.parameters()]))
# karray == df[0,:, keymap['longname']]
        nlayers = len(layernames)
        # print (f"Number of layers: {nlayers}")
    else:
        nlayers = 1

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


    client_models: dict[int, nn.Module] = {}
    client_deltas: dict[int, nn.Module] = {}
    client_train_loaders = {}
    client_val_loaders = {}
    # if 'fedopt' in cfg.strategy:
    agg_gradient = [torch.zeros_like(param.data) for param in model_base.parameters()]

    for i in range(cfg.num_clients):
        client_models[i] = deepcopy(model_base)
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

    global_model.to(cfg.device)
    data_lengths = [len(client_sets[i]) for i in range(cfg.num_clients)]

    agg_weights = 1/cfg.num_clients*np.ones((cfg.num_clients, nlayers))  # Initialize aggregation weights
    print(f"Aggregation Weights Shape: {agg_weights.shape}")
    # fedavg_weights = np.array(data_lengths) / np.sum(data_lengths)

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

    cfg.save_config()

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    entropies = np.full(cfg.num_clients, np.nan, dtype=float)
    smooth_entropies = np.full(cfg.num_clients, np.nan, dtype=float)

    total_time = []
    for r in range(cfg.num_rounds):
        start_proc_time = time.process_time()

        print(f"Round {r + 1}/{cfg.num_rounds}")
        logger.curr_round.append(r)

        client_params = {}
        temp_weights = agg_weights.copy()

        selected_clients = np.random.choice(
            range(cfg.num_clients), 
            size=num_selected_clients, 
            replace=False
        )
        selected_mask = np.zeros(cfg.num_clients, dtype=bool)
        selected_mask[selected_clients] = True
        for i, client in client_models.items():
            if not selected_mask[i]:
                continue

            # start_prep_time = time.process_time()
            
            print(f"Training client {i}/{cfg.num_clients}")
            client.to(cfg.device)
            client.train()  # Set the client model to training mode

            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
                logger.lr_summary.append(lr)

            else:
                lr = cfg.lr

            optimizer = cfg.optimizer_partial(
                client.parameters(),
                lr=lr,
            )
            # end_prep_time = time.process_time()
            # print(f"Client {i} preparation CPU time: {end_prep_time - start_prep_time} seconds")
            # start_train_time = time.process_time()

            for e in range(cfg.num_epochs):
                # Your GPU operations
                # start_event.record()

                client, losses, acc, bacc = train_one_epoch_model(client, client_train_loaders[i], optimizer, cfg.criterion, cfg.device
                )
                # end_event.record()
                # torch.cuda.synchronize() # Wait for all GPU operations to complete
                # gpu_execution_time_ms = start_event.elapsed_time(end_event) # Time in milliseconds
                # print(f"GPU execution time in training: {gpu_execution_time_ms / 1000:.4f} seconds")
    
                logger.client_trn_losses[i].extend(losses)
                logger.client_trn_accs[i].append(acc)
                logger.client_trn_baccs[i].append(bacc)

            end_train_time = time.process_time()
            # print(f"Client {i} training CPU time: {end_train_time - start_train_time} seconds")


            start_eval_time = time.process_time()
            # start_event.record()

            test_acc, test_bacc, f1_wtd, f1_micro,  test_losses = evaluate_model(client, test_loader, cfg.criterion, cfg.device)
            # end_event.record()
            # torch.cuda.synchronize() # Wait for all GPU operations to complete
            # gpu_execution_time_ms = start_event.elapsed_time(end_event) # Time in milliseconds
            # print(f"GPU execution time in evaluation: {gpu_execution_time_ms / 1000:.4f} seconds")

            end_eval_time = time.process_time()
            # print(f"Client {i} evaluation CPU time: {end_eval_time - start_eval_time} seconds")

            start_cl_add_time = time.process_time()

            logger.client_test_accs[i].append(test_acc)
            logger.client_test_baccs[i].append(test_bacc)
            logger.client_test_f1wtd[i].append(f1_wtd)
            logger.client_test_f1mic[i].append(f1_micro)
            logger.client_test_losses[i].extend(test_losses)

            # Find the difference between the client model and the global model
            delta_model = client.state_dict()
            for k, v in global_model.named_parameters():
                delta_model[k] = delta_model[k] - v.data

            client_deltas[i].load_state_dict(delta_model)
            # client_deltas[i].to("cpu")

            # cww = ww.WeightWatcher(client)
            cww = ww.WeightWatcher(client_deltas[i], log_level="ERROR")
            analysis = cww.analyze()
            summary = cww.get_summary()
            logger.client_ww_details[i].append(analysis.to_numpy()) # type: ignore
            logger.client_ww_summary[i].append(list(summary.values()))
            logger.client_entropy_ll_sclr[i].append(analysis['entropy'].to_numpy()[-1]) # type: ignore

            if 'lwise' in cfg.method:
                avl_layers = analysis['longname'].to_numpy() # type: ignore
                entropies = analysis['entropy'].to_numpy() # type: ignore

                for j, l in enumerate(layernames):
                    if '.weight' in l:
                        k1 = l.removesuffix('.weight')
                    elif '.bias' in l:
                        k1 = l.removesuffix('.bias')
                    else:
                        k1 = l
                    if l in avl_layers:
                        # agg_weights[i,j] = entropies[k1]
                        temp_weights[i,j] = entropies[k1]
                    else:
                        if 'mode_ll' in cfg.method:
                            temp_weights[i,j] = entropies[-1]
                            # agg_weights[i,j] = entropies[-1]
            else:
                # agg_weights[i,:] = analysis['entropy'].to_numpy()[-1]
                temp_weights[i,:] = analysis['entropy'].to_numpy()[-1] # type: ignore
                # logger.client_weights_sclr[i].append(temp_weights[i,:].copy())

            client_params[i] = client.state_dict()

            # end_cl_add_time = time.process_time()
            # print(f"Client {i} aggregation data preparation CPU time: {end_cl_add_time - start_cl_add_time} seconds")

        start_agg_time = time.process_time()

        tick = time.time()

        # for i in range(cfg.num_clients):
        #     pure_entropy = layer_spectral_entropy(client_deltas[i].state_dict()[f"{cfg.layer_prefix}.weight"],).item()
        # Normalize the aggregation weights
        if 'softmax_weights' in cfg.method:
            temp_weights = np.exp(cfg.alpha*temp_weights)
    
        # Momentum
        agg_weights[selected_clients] = (1 - cfg.mu) * temp_weights[selected_clients] + cfg.mu * agg_weights[selected_clients]

        # Normalize
        # agg_weights =  agg_weights /(agg_weights.sum(axis=0, keepdims=True))
        agg_weights_round = np.zeros_like(agg_weights)
        agg_weights_round[selected_clients] = agg_weights[selected_clients] / (
            agg_weights[selected_clients].sum(axis=0, keepdims=True) + 1e-12
        )
        if 'lwise' in cfg.method:
            for i in range(cfg.num_clients):
                logger.client_weights_lyr[i].append(agg_weights[i,:].copy())
        else:
            for i in range(cfg.num_clients):
                logger.client_weights_sclr[i].append(agg_weights[i,0].copy())


        # Entropy averaging
        
        for l, (key, param) in enumerate(global_model.named_parameters()):
            if "fedopt" in cfg.strategy:
                agg_delta = torch.zeros_like(param.data)
                for cid, delta_model in client_deltas.items():
                    if cid not in selected_clients:
                        continue
                    csd = delta_model.state_dict()
                    if 'lwise' in cfg.method:
                        agg_delta.add_(agg_weights_round[cid,l] * csd[key].data)
                    else:
                        agg_delta.data.add_(agg_weights_round[cid,0] * csd[key].data)
                param.data.add_(agg_delta)
                agg_gradient[l] = agg_delta
            else:
                temp_parameter = torch.zeros_like(param.data)
                for cid, c_state in client_params.items():
                    if cid not in selected_clients:
                        continue
                    if 'lwise' in cfg.method:
                        temp_parameter.data.add_(agg_weights_round[cid,l] * c_state[key].data)
                    else:
                        temp_parameter.data.add_(agg_weights_round[cid,0] * c_state[key].data)

                param.data.copy_(temp_parameter)

        tock = time.time()
        # if r >0 and r < 11:
        #     total_time.append(tock - tick)
        #     avg_time, std_time = np.mean(total_time), np.std(total_time)
        #     print(f"Average time over rounds 1-{r}: {avg_time:.6f} +- {std_time:.6f} seconds (wall)")

        if scheduler is not None:
            scheduler.step()
        # global_model.load_state_dict(global_state_dict)        # use recalibrate_bn if resnet models
        if isinstance(global_model, ResNet):
            global_model = recalibrate_bn(
                global_model, test_loader, cfg.batch_size, num_batches=5, device=cfg.device
            )
        # global_model = recalibrate_bn(global_model, test_loader, num_batches=5)
        global_acc, global_bacc, global_f1_wtd, global_f1_micro, global_test_loss = evaluate_model(
            global_model, test_loader, cfg.criterion, cfg.device
        )

        logger.server_accs.append(global_acc)
        logger.server_baccs.append(global_bacc)
        logger.server_f1wtd.append(global_f1_wtd)
        logger.server_f1mic.append(global_f1_micro)
        logger.server_test_losses.extend(global_test_loss)
        print(f"Global Model - Round {r + 1} - Test Accuracy: {global_acc}")
        print("\n")

        gww = ww.WeightWatcher(global_model, log_level="ERROR")
        gdetails = gww.analyze()
        summary = gww.get_summary()
        logger.server_ww_details.append(gdetails.to_numpy()) # type: ignore
        logger.server_ww_summary.append(list(summary.values()))

        global_state_dict = global_model.state_dict()

        ### send the global model to all clients
        for cid, client in client_models.items():
            # client_state_dict = client.state_dict()
            if cfg.rewards == "interpolation":
                if agg_weights.shape[1] > 1:
                    agg_weights_cid = torch.tensor(agg_weights[cid,:], device=cfg.device, dtype=torch.float32).mean(1)
                else:
                    agg_weights_cid = torch.tensor(agg_weights[cid], device=cfg.device, dtype=torch.float32)
                interpolation_rewards(global_model.parameters(), client.parameters(), coeff=agg_weights_cid)
            elif cfg.rewards == "sparsification":
                if agg_weights.shape[1] > 1:
                    agg_weights_cid = torch.tensor(agg_weights[cid,:], device=cfg.device, dtype=torch.float32).mean(1)
                else:
                    agg_weights_cid = torch.tensor(agg_weights[cid], device=cfg.device, dtype=torch.float32)
                if 'fedopt' in cfg.strategy:
                    sparsification_gradient_rewards(client.parameters(), agg_gradient, coeff=agg_weights_cid, beta=1.0)
                else:
                    sparsification_param_rewards(client.parameters(), global_model.parameters(), coeff=agg_weights_cid, beta=1.0)
            elif cfg.rewards == "none":
                no_rewards(global_model.parameters(), client.parameters())
            else:
                raise ValueError(f"Unknown rewards method: {cfg.rewards}")
            
            # for key, param in client.named_parameters():
            #     # for key, param in client.state_dict().items():
            #     # param.data = global_state_dict[key].data.copy_()
            #     param.data.copy_(global_state_dict[key].data)

        end_cpu_time = time.process_time()
        # start_agg_time = time.process_time()
        # print(end_cpu_time - start_agg_time, "seconds of CPU time used in aggregation.")

        # logger.proc_times.append(end_cpu_time - start_proc_time)
        # print(end_cpu_time - start_proc_time, "seconds of CPU time used.")

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
    parser.add_argument("-s", "--strategy", type=str, default="spectralfed")
    parser.add_argument( '-d',"--debug", action="store_true")
    parser.add_argument("--split", type=str, default="iid")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="tf_cnn")
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--reward", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    # for free_rider_idx in [0,1,2,3,4]:
    #     print("=========================================")
    #     print(f"Running for free_rider_idx: {free_rider_idx}")
    free_rider_idx = 1  # You can change this value as needed
    cfg = SpectralFedConfig(
        seed=args.seed,
        # device=torch.device(auto_configure_device()),
        strategy=args.strategy,
        # strategy="spectralfed",
        optimizer=optim.SGD,
        criterion=nn.CrossEntropyLoss(),
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        lr=0.1,
        batch_size=64,
        # dataset_name="fedisic",
        dataset_name=args.dataset,
        freeze_cfg=False,
        num_rounds=200,
        num_clients=10,
        participation_rate=0.5,
        # model_name="vit_b_16_pret",
        # model_name="tf_cnn",
        model_name=args.model,
        # model_name="resnet18",
        split_cfg={"min_labels": 1}, 
        # split="dirichlet",
        # split_cfg={"alpha": 1.0},
        # split = 'iid',
        # split_cfg={"free_rider_idx": free_rider_idx, "free_rider_actual_size": 10},
        split=args.split,
        # split='free_rider',
        # method="linear_weights_lwise",
        method=f"linear_weights_{args.suffix}",
        # rewards="none",
        rewards=args.reward,
        # rewards="interpolation",
        # rewards="sparsification",
        # method="softmax_weights",
        # method=f"free_rider_{free_rider_idx}",
        # expt_obj=f"free_rider_{args.suffix}",
        alpha=1.0,
        # mu=0.9,
        mu=args.mu,
        notes="linear wts with momentum",

    )
        # print(f"Dirichlet split with alpha: {cfg.split_cfg['alpha']}")
    if args.debug:
        cfg.num_rounds = 12
        cfg.scheduler = None
    
# python spectralfed.py -s spectralfed --split natural --dataset fedisic --model vit_b_16_pret --mu=0.9 --reward none --suffix mu0.9_new
    # exit(0)

    cfg.print_summary()

    run_spectralfed(cfg)
