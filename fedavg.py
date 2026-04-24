import os
import yaml
from multiprocessing import Pool, set_start_method
import argparse
from copy import deepcopy
import time
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pandas import json_normalize
import weightwatcher as ww
from pathlib import Path


# For CIFAR-10
from torchvision.models import resnet18, resnet50, ResNet

from data import get_fl_datasets_and_model

from trainutils import train_one_epoch_model, evaluate_model, adapt_model_last_layer

from utils import (
    append_to_ledger,
    setup_output_dirs,
    set_seed,
    auto_configure_device,
    json_dump,
    yaml_dump,
    FLLogger,
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
class FedAvgLogger(FLLogger):
    client_ww_details: dict[int, list] = field(default_factory=dict)
    client_ww_summary: dict[int, list] = field(default_factory=dict)
    server_ww_details: list = field(default_factory=list)
    server_ww_summary: list = field(default_factory=list)

    def __post_init__(self):
        # Initialize client and server weight watcher details
        self.client_ww_details = {i: [] for i in range(self.num_clients)}
        self.client_ww_summary = {i: [] for i in range(self.num_clients)}
        super().__post_init__()

def run_fedavg(cfg: Config):
    start_time = time.time()

    logger = FedAvgLogger(
        num_clients=cfg.num_clients, output_dir=cfg.output_dir, use_tensorboard=True
    )
    if cfg.resumed:
        logger.resume_logger(cfg.output_dir)
        os.environ['OUT_DIR'] = str(cfg.output_dir)
        print("Resumed logger from existing directory.")
        start_round = logger.curr_round[-1] + 1
        print(f"Current round: {logger.curr_round[-1]+1}")
        # input("Press Enter to continue...")
        if 'dirichlet' in cfg.split:
            cfg.split_cfg['alpha'] = float(cfg.split.split('_')[-1])
            cfg.split = 'dirichlet'

    else:
        start_round = 0

    client_sets, test_dataset, model_base = get_fl_datasets_and_model(
        dataset_name=cfg.dataset_name,
        model_name=cfg.model_name,
        split_name=cfg.split,
        num_clients=cfg.num_clients,
        split_config=cfg.split_cfg
    )

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )
    num_selected_clients = max(1, int(cfg.participation_rate * cfg.num_clients))

    client_models: dict[int, nn.Module] = {}
    client_deltas: dict[int, nn.Module] = {}
    client_train_loaders = {}
    client_val_loaders = {}

    for i in range(cfg.num_clients):
        client_models[i] = deepcopy(model_base)
        if cfg.resumed:
            client_models[i].load_state_dict(
                torch.load(cfg.output_dir / f"model_{i}.pth")
            )
        client_deltas[i] = deepcopy(model_base)
        client_deltas[i].to(cfg.device)
        client_train_loaders[i] = DataLoader(
            client_sets[i], batch_size=cfg.batch_size, shuffle=True, num_workers=4  # type: ignore
        )

    global_model = deepcopy(model_base)
    if cfg.resumed:
        global_model.load_state_dict(torch.load(cfg.output_dir / "global_model.pth"))
    global_model.to(cfg.device)

    if "uni" in cfg.strategy:
        # Use uniform weights for FedAvg-Uniform
        fedavg_weights = np.ones(cfg.num_clients) / cfg.num_clients
    else:
        data_lengths = [len(client_sets[i]) for i in range(cfg.num_clients)]
        fedavg_weights = np.array(data_lengths) / np.sum(data_lengths)

    if cfg.resumed:
        start_lr = logger.lr_summary[-1]
        print(f"Resumed learning rate: {start_lr}")
    else:
        start_lr = cfg.lr
    

    server_optimizer = cfg.optimizer_partial(global_model.parameters(), lr=start_lr)

    scheduler = (
        cfg.scheduler_partial(server_optimizer)
        if isinstance(cfg.scheduler, type)
        else None
    )

    # Logging

    json_dump(
        server_optimizer.__dict__["defaults"], cfg.output_dir / "optimizer_cfg.json"
    )
    if scheduler is not None:
        scheduler_attributes = scheduler.__dict__.copy()
        scheduler_attributes.pop(
            "optimizer", None
        )  # Remove the optimizer from the dict
        json_dump(scheduler_attributes, cfg.output_dir / "scheduler_cfg.json")
        torch.save(scheduler.state_dict(), cfg.output_dir / "scheduler.pth")

    cfg.save_config()

    for r in range(start_round, cfg.num_rounds):
        print(f"Round {r + 1}/{cfg.num_rounds}")
        logger.curr_round.append(r)
        client_params = {}
        # client_param_deltas = {}
        agg_weights = {}
        selected_clients = np.random.choice(
            range(cfg.num_clients), 
            size=num_selected_clients, 
            replace=False
        )
        fedavg_weights_round = fedavg_weights.copy()
        fedavg_weights_round[selected_clients] = fedavg_weights_round[selected_clients] / np.sum(fedavg_weights_round[selected_clients])  # re-normalize
        print(f"Selected clients: {selected_clients}")
        # print(f"Normalized FedAvg weights: {fedavg_weights_round}")

        for i, client in client_models.items():
            if i not in selected_clients:
                continue
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
            for e in range(cfg.num_epochs):

                client, losses, acc, bacc = train_one_epoch_model(
                    client,
                    client_train_loaders[i],
                    optimizer,
                    cfg.criterion,
                    cfg.device,
                )
                logger.client_trn_losses[i].extend(losses)
                logger.client_trn_accs[i].append(acc)
                logger.client_trn_baccs[i].append(bacc)

            test_acc, test_bacc, f1_wtd, f1_micro,  test_losses = evaluate_model(client, test_loader, cfg.criterion, cfg.device)

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

            # cww = ww.WeightWatcher(client)
            cww = ww.WeightWatcher(client_deltas[i], log_level="ERROR")
     
            # analysis = cww.analyze()
            # summary = cww.get_summary()
            # logger.client_ww_details[i].append(analysis.to_numpy())
            # logger.client_ww_summary[i].append(list(summary.values()))
            # agg_weights[i] = analysis['entropy'].iloc[-1]

            client_params[i] = client.state_dict()

        for key, param in global_model.named_parameters():
            # for key, param in global_model.state_dict().items():
            if "fedavg" in cfg.strategy:
                temp_parameter = torch.zeros_like(param.data)

                for cid, c_state in client_params.items():
                    if cid not in selected_clients:
                        continue
                    temp_parameter.data.add_(fedavg_weights_round[cid] * c_state[key].data)
                param.data.copy_(temp_parameter)
                
            elif "fedopt" in cfg.strategy:
                temp_delta = torch.zeros_like(param.data)

                for cid, delta_model in client_deltas.items():
                    if cid not in selected_clients:
                        continue
                    csd = delta_model.state_dict()
                    # print(f"csd device: {csd[key].device}, type: {type(csd[key])}")
                    temp_delta.data.add_(fedavg_weights_round[cid] * csd[key].data)
                param.data.add_(temp_delta)

            else:
                raise ValueError(f"Unknown strategy: {cfg.strategy}")


        if scheduler is not None:
            scheduler.step()
        # global_model.load_state_dict(global_state_dict)
        # use recalibrate_bn if resnet models
        if isinstance(global_model, ResNet):
            global_model = recalibrate_bn(global_model, test_loader, cfg.batch_size, num_batches=5, device=cfg.device)

        global_acc, global_bacc, global_f1w, global_f1mic, global_test_loss = evaluate_model(global_model, test_loader, cfg.criterion, cfg.device)

        logger.server_accs.append(global_acc)
        logger.server_baccs.append(global_bacc)
        logger.server_f1wtd.append(global_f1w)
        logger.server_f1mic.append(global_f1mic)
        logger.server_test_losses.extend(global_test_loss)
        print(f"Global Model - Round {r + 1} - Test Accuracy: {global_acc}")
        print("\n")

        # gww = ww.WeightWatcher(global_model)
        # gdetails = gww.analyze()
        # summary = gww.get_summary()
        # logger.server_ww_details.append(gdetails.to_numpy())
        # logger.server_ww_summary.append(list(summary.values()))

        global_state_dict = global_model.state_dict()

        ### send the global model to all clients
        for cid, client in client_models.items():
            # client_state_dict = client.state_dict()
            for key, param in client.named_parameters():
                # for key, param in client.state_dict().items():
                # param.data = global_state_dict[key].data.copy_()
                param.data.copy_(global_state_dict[key].data)


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
            elapsed_time = time.time() - start_time
            # run_summary = logger.generate_summary()
            # run_summary["elapsed_time"] = np.round(elapsed_time, 2)

            # Save the run summary
            # json_dump(run_summary, cfg.output_dir / "run_summary.json")

            # all_summary = cfg.main_summary.copy()
            # all_summary.update(run_summary)  # type: ignore

            # append_to_ledger(all_summary)
        # Finetuning for client accuracies
    for i, client in client_models.items():
        print(f"Finetuning client {i} model on its local data")
        client.to(cfg.device)
        
        client, losses, acc, bacc = train_one_epoch_model(
            client,
            client_train_loaders[i],
            optimizer=cfg.optimizer_partial(
                client.parameters(),
                lr=start_lr,
            ),
            criterion=cfg.criterion,
            device=cfg.device,
        )
        test_acc, test_bacc, f1_wtd, f1_micro,  test_losses = evaluate_model( client, test_loader, cfg.criterion, cfg.device
        )
        print(f"Client {i} - Finetuned Test Accuracy: {test_acc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--strategy", type=str, default="fedavg_uni")
    parser.add_argument( '-d',"--debug", action="store_true")
    parser.add_argument("--split", type=str, default="iid")
    parser.add_argument("--dataset", type=str, default="cifar10")
    # parser.add_argument("--dataset", type=str, default="fashionmnist")
    parser.add_argument("--model", type=str, default="tf_cnn")
    # parser.add_argument("--model", type=str, default="mlpnet")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default="")  # Path to checkpoint to resume from
    args = parser.parse_args()

    if args.resume_from != "":
        resumed_run = True
    else:
        resumed_run = False
    # Create a default configuration
    cfg = Config(
        seed=args.seed,
        strategy=args.strategy,
        # strategy="fedavg_uni",
        # strategy="fedavg",
        optimizer=optim.SGD,
        criterion=nn.CrossEntropyLoss(),
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        lr=0.1,
        batch_size=64,
        split=args.split,
        # split="dirichlet",
        # split_cfg={"alpha": 0.1},
        split_cfg={"min_labels": 1}, 
        # model_name="tf_cnn",
        # model_name="mlpnet",
        model_name=args.model,
        dataset_name=args.dataset,
        expt_obj=f"{args.suffix}",
        freeze_cfg=False,
        num_rounds=200,
        num_clients=10,
        participation_rate=0.5,
        # freeze_cfg=True,
        resumed=resumed_run,
        notes=f"{args.dataset} {args.model} {args.strategy} {args.split} {args.suffix}",
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
    run_fedavg(cfg)



    # Run the federated averaging algorithm with the configuration
