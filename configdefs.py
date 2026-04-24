from dataclasses import dataclass, field, asdict, MISSING
import typing as t
from functools import partial
import os
from pathlib import Path
import yaml
import json

import torch
from utils import auto_configure_device, set_seed, setup_output_dirs
from pandas import json_normalize

## GLOBAL PATH DECLARATIONS
# DATA_PATH = "/home/asim.ukaye/fed_learning/simplefl/data"
# DATA_PATH = Path("/home/asim.ukaye/fed_learning/spectralfed/data")
# OUTPUT_PATH =Path( "/home/asim.ukaye/fed_learning/datasets/flbase_output")
HOME_DIR = Path(os.getcwd())
OUTPUT_PATH = HOME_DIR / "output"
DATA_PATH = HOME_DIR / "data"
 

BIG_DATA_PATH = Path(os.environ.get("BIG_DATA_PATH", "/home/asim.ukaye/fed_learning/datasets"))  # type: ignore
os.environ['BIG_DATA_PATH'] = str(BIG_DATA_PATH)

# SEED = 42
from torch.nn import (
    Module,
    CrossEntropyLoss,
    NLLLoss,
    MSELoss,
    L1Loss,
    BCELoss,
    BCEWithLogitsLoss,
    CTCLoss,
    KLDivLoss,
    MultiMarginLoss,
    SmoothL1Loss,
    TripletMarginLoss,
    CosineEmbeddingLoss,
    PoissonNLLLoss,
    HuberLoss,
    HingeEmbeddingLoss,
)
from torch.optim import Optimizer, SGD, Adam, AdamW, Adadelta, Adagrad, RMSprop
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
)

## Root level module. Should not have any dependencies on other modules except utils

from utils import auto_configure_device

OPTIMIZER_MAP: dict[str, type[Optimizer]] = {
    "Adam": Adam,
    "SGD": SGD,
    "AdamW": AdamW,
    "Adagrad": Adagrad,
    "Adadelta": Adadelta,
    "RMSprop": RMSprop,
}

OPTIM_DEFAULTS = {
    SGD: {},
    Adam: {"betas": (0.9, 0.999), "eps": 1e-8},
    AdamW: {"weight_decay": 1e-4},
}
LRSCHEDULER_MAP = {
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CyclicLR": CyclicLR,
    "OneCycleLR": OneCycleLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
}
SCHEDULER_DEFAULTS = {
    StepLR: {"step_size": 30, "gamma": 0.1},
    MultiStepLR: {"milestones": [30, 80], "gamma": 0.1},
    ExponentialLR: {"gamma": 0.95},
    CosineAnnealingLR: {"eta_min": 1e-6},
    ReduceLROnPlateau: {"mode": "min", "factor": 0.1, "patience": 10},
    CyclicLR: {"base_lr": 0.001, "max_lr": 0.01, "step_size_up": 2000},
    OneCycleLR: {"max_lr": 0.01, "total_steps": 10000},
    CosineAnnealingWarmRestarts: {"T_0": 10, "T_mult": 2},
}
LOSS_MAP = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "NLLLoss": NLLLoss,
    "MSELoss": MSELoss,
    "L1Loss": L1Loss,
    "BCELoss": BCELoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "CTCLoss": CTCLoss,
    "KLDivLoss": KLDivLoss,
    "MultiMarginLoss": MultiMarginLoss,
    "SmoothL1Loss": SmoothL1Loss,
    "HuberLoss": HuberLoss,
    "TripletMarginLoss": TripletMarginLoss,
    "HingeEmbeddingLoss": HingeEmbeddingLoss,
    "CosineEmbeddingLoss": CosineEmbeddingLoss,
    "PoissonNLLLoss": PoissonNLLLoss,
}


@dataclass
class Config:
    participation_rate: float = 1.0  # fraction of clients participating each round
    timestamp: int = 0
    strategy: str = "fedavg"
    optimizer: type[Optimizer] = SGD
    criterion: Module = field(default_factory=CrossEntropyLoss)
    scheduler: type[LRScheduler] | None = None
    lr: float = 0.1
    batch_size: int = 64
    num_epochs: int = 1
    device: torch.device | None = None
    split: str = "iid"
    split_cfg: dict = field(default_factory=dict)
    model_name: str = "resnet18"
    dataset_name: str = "cifar10"
    num_clients: int = 5
    num_rounds: int = 200
    notes: str = ""
    expt_obj: str = "baseline"
    seed: int = 42
    output_dir: Path = OUTPUT_PATH
    error: int = 0
    hostname: str = os.uname().nodename
    pid: int = 0
    resumed: bool = False
    freeze_cfg: bool = True

    def __post_init__(self):
        ## Define internal config variables here
        # self.use_wandb = True
        self.pid = os.getpid()
        if self.device is None:
            self.device = torch.device(auto_configure_device())
        os.environ["ACTIVE_DEVICE"] = str(self.device)

        if not self.resumed:

            run_name = f"{self.split}-{self.expt_obj}"

            self.output_dir, self.timestamp = setup_output_dirs(
                self.strategy, self.dataset_name, self.model_name, sub_dir=run_name)
            if self.freeze_cfg:
                self.freeze_dataset_cfgs()
            set_seed(self.seed)

            self.main_summary = asdict(self)
            self.main_summary.pop("freeze_cfg", None)
            self.main_summary["device"] = str(self.device)
            self.main_summary["output_dir"] = str(self.output_dir)

            self.main_summary["optimizer"] = self.optimizer.__name__
            self.main_summary["criterion"] = self.criterion.__class__.__name__
            self.main_summary["scheduler"] = (
                self.scheduler.__name__ if isinstance(self.scheduler, type) else "None"
            )

            if "dirichlet" in self.split:
                alpha_list = self.split.split("_")
                if len(alpha_list) > 1:
                    self.split_cfg = {"alpha": float(alpha_list[-1])}
                else:
                    self.split_cfg = {"alpha": 1.0}
                self.split = "dirichlet"
            self.main_summary["split_cfg"] = self.split_cfg
            self.setup_partials()

        self.eval_batch_size = 64


    def setup_partials(self):
        self.optimizer_partial = partial(self.optimizer, **OPTIM_DEFAULTS.get(self.optimizer.__name__, {}))  # type: ignore
        if isinstance(self.scheduler, type):
            if self.scheduler == CosineAnnealingLR:
                self.scheduler_partial = partial(
                    self.scheduler, **{"T_max": self.num_rounds, "eta_min": 1e-6}
                )
            elif self.scheduler == MultiStepLR:
                self.scheduler_partial = partial(
                    self.scheduler,
                    milestones=[self.num_rounds // 2, int(self.num_rounds * 0.75)],
                    **SCHEDULER_DEFAULTS.get(self.scheduler.__name__, {}),
                )
            else:
                self.scheduler_partial = partial(self.scheduler, **SCHEDULER_DEFAULTS.get(self.scheduler.__name__, {}))  # type: ignore

    def load_config(self, path: Path) -> None:
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
            self.main_summary = cfg_dict.copy()

        for key, value in cfg_dict.items():
            if key in ["device", "pid", 'hostname']:
                if hasattr(self, key):
                    curr_value = getattr(self, key)
                    self.main_summary[key] = curr_value
                continue
            if hasattr(self, key):
                if key in ["optimizer", "criterion", "scheduler", 'output_dir']:
                    if key == "optimizer":
                        setattr(self, key, OPTIMIZER_MAP.get(value, SGD))  # type: ignore
                    elif key == "scheduler":
                        if value == "None":
                            setattr(self, key, None)
                        else:
                            setattr(self, key, LRSCHEDULER_MAP.get(value, None))  # type: ignore
                    elif key == "criterion":
                        setattr(self, key, LOSS_MAP.get(value, CrossEntropyLoss)())  # type: ignore
                    elif key == "output_dir":
                        setattr(self, key, Path(value))
                else:
                    setattr(self, key, value)

        self.main_summary['device'] = str(self.device)
        self.resumed = True
        set_seed(self.seed)
        self.setup_partials()


    def print_summary(self):
        print("Configuration Summary:")
        print(
            yaml.dump(self.main_summary, allow_unicode=True, default_flow_style=False)
        )

    def save_config(self) -> None:
        """Save the configuration to a YAML file."""
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(self.main_summary, f, indent=4, allow_unicode=True)

    def freeze_dataset_cfgs(self) -> None:
        """Freeze the dataset and model configurations."""
        if self.dataset_name == "fedisic":
            self.num_clients = 6
            self.num_rounds = 300
            self.lr = 0.001
        elif self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            self.num_clients = 5
            self.num_rounds = 200
            self.lr = 0.1
        elif self.dataset_name == "mnist":
            self.num_clients = 5
            self.num_rounds = 30
            self.lr = 0.1
        elif self.dataset_name == "fashionmnist":
            self.num_clients = 5
            self.num_rounds = 100
            self.lr = 0.1
        elif self.dataset_name == "femnist":
            self.num_clients = 100
            self.num_rounds = 100
            self.lr = 0.1
        else:
            self.num_clients = 5
            self.num_rounds = 200
            self.lr = 0.1
