import logging
import os
import random
import subprocess
from io import StringIO
import unicodedata
import re
import time
from pathlib import Path
import sys
from dataclasses import dataclass, field, asdict, InitVar
import json
import yaml

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from wonderwords import RandomWord
from torch.utils.tensorboard.writer import SummaryWriter
from filelock import FileLock

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"[SEED] Simulator global seed is set to: {seed}!")


# Naming functions


def get_client_train_val_indices(clients_indices: list, train_ratio=0.9):
    client_train_val_indices = {}
    for i, c_idx in enumerate(clients_indices):
        train_indices = np.random.choice(
            c_idx, int(train_ratio * len(c_idx)), replace=False
        )
        val_indices = np.setdiff1d(c_idx, train_indices)
        client_train_val_indices[i] = (train_indices, val_indices)
    return client_train_val_indices


def make_random_name():
    r = RandomWord()
    name = "-".join(
        [
            r.word(
                word_min_length=3,
                word_max_length=7,
                include_parts_of_speech=["adjective"],
            ),
            r.word(
                word_min_length=5, word_max_length=7, include_parts_of_speech=["noun"]
            ),
        ]
    )
    return name


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def generate_client_ids(num_clients):
    return [f"{idx:04}" for idx in range(num_clients)]


# Logging functions
def json_dump(obj: dict, fname: Path, indent=4) -> None:
    """Convert an object to a JSON string with indentation."""
    with open(fname, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def yaml_dump(obj: dict, fname: Path, indent=4) -> None:
    """Convert an object to a YAML string with indentation."""
    with open(fname, "w") as f:
        yaml.dump(obj, f, indent=indent, allow_unicode=True)


def append_to_ledger(in_dict: dict, outpath="output") -> None:
    """Convert an object to an csv file."""
    # Open Ledger file or create it if it doesn't exist
    # assert 'timestamp' in obj, "Object must contain a 'timestamp' key"
    # if not os.path.exists(Path(outpath) / "ledger.xlsx"):
    #     df = pd.DataFrame(columns=list(obj.keys() - {'timestamp'}), index=['timestamp'])
    # else:
    #     df = pd.read_excel(Path(outpath) / "ledger.xlsx", engine='openpyxl')
    # # Append the new object to the DataFrame
    # obj_flat = pd.json_normalize(obj)
    # df = pd.concat([df, obj_flat], ignore_index=True)

    # df.to_excel(Path(outpath) / "ledger.xlsx", engine='openpyxl')

    data_dict = in_dict.copy()
    # file = Path("output") / "ledger.csv"
    file = Path(".") / "ledger.csv"
    timestamp = data_dict.pop("timestamp")
    # pd_time = pd.to_datetime(timestamp, format="%y%m%d-%H%M%S")
    lctime = time.localtime(timestamp * 1e-9)
    data_dict["date"] = time.strftime("%Y-%m-%d", lctime)
    data_dict["time"] = time.strftime("%H-%M-%S", lctime)
    data_dict["error"] = 0
    new_df = pd.DataFrame([data_dict], index=[timestamp])
    new_df.index.name = "timestamp"

    with FileLock(file.with_suffix(".lock"), timeout=10):
        if file.exists():
            # Load existing
            # df = pd.read_excel(file, index_col=0, sheet_name="main")
            try:
                df = pd.read_csv(file, index_col=0)
            except pd.errors.EmptyDataError:
                print(f"File {file} is empty. passing this round.")
                return

            # Add new columns if needed
            for col in new_df.columns:
                if col not in df.columns:
                    df[col] = pd.NA  # Or np.nan

            # Add missing columns in new_row too (in case the file has columns that new_row does not)
            for col in df.columns:
                if col not in new_df.columns:
                    new_df[col] = pd.NA
            # Update or append
            df.loc[timestamp] = new_df.iloc[0]
        else:
            # Create new
            df = new_df

        # Save back to Excel
        df.sort_index(inplace=True)
        df.to_csv(file, index=True)
        # df.to_excel(file, sheet_name="main")

def pick_new_name(output_dir: Path) -> str:
    """Pick a new random name that does not exist in the output directory."""
    while True:
        random_name = make_random_name()
        new_dir = f"{output_dir.name}-{random_name}"
        if not (output_dir.parent / new_dir).exists():
            return new_dir
        else:
            print(f"Directory {new_dir} already exists. Picking a new name...")

def setup_output_dirs(
    strategy: str,
    dataset_name: str,
    model_name: str,
    sub_dir: str = "",
    output_path="output",
) -> tuple[Path, int]:

    # Setup the run
    # run_type = f"{strategy:.7}-{cfg.dataset.name}-{cfg.split.name:.7}"
    main_dir = f"{strategy}-{dataset_name}-{model_name}"

    main_dir = slugify(main_dir, allow_unicode=True)

    timestamp_ns = time.time_ns()

    run_date_time = time.strftime("%y%m%d-%H%M%S", time.localtime(timestamp_ns * 1e-9))
    if sub_dir == "":
        random_name = make_random_name()
        sub_dir = f"{run_date_time}-{random_name}"
    else:
        sub_dir = f"{run_date_time}-{sub_dir}"

    sub_dir = slugify(sub_dir, allow_unicode=True)
    output_base = Path(output_path) / main_dir
    output_base.mkdir(parents=True, exist_ok=True)
    output_dir = output_base / sub_dir
    if output_dir.exists():
        # output_dir = output_base / f"{sub_dir}-{timestamp_ns}"
        output_dir = output_base / f"{pick_new_name(output_dir)}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Logging to {output_dir}")
    # os.chdir(output_dir)
    os.environ["OUT_DIR"] = str(output_dir)

    return output_dir, timestamp_ns


def get_wandb_run_id(root_dir=".") -> str:
    list_dir = os.listdir(root_dir + "/wandb/latest-run")
    for list_item in list_dir:
        if ".wandb" in list_item:
            run_id = list_item.split("-")[-1].removesuffix(".wandb")
            print(f"Found run_id: {run_id}")
            return run_id
    raise FileNotFoundError(
        f"No wandb run_id found in {root_dir}/wandb/latest-run. Please check if the directory exists."
    )


def setup_logging(level=logging.INFO, add_file_handler=False):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    verbose_formatter = logging.Formatter(
        "%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    if add_file_handler:
        file_handler = logging.FileHandler("flbase.log")
        file_handler.mode = "a"
        file_handler.setLevel(level)
        file_handler.setFormatter(verbose_formatter)
        root_logger.addHandler(file_handler)


def init_client_log_list(nclients: int) -> list:
    """Initialize a list of empty lists for each client."""
    return [[] for _ in range(nclients)]

def get_nested_value(nested_dict: dict, keys_list:list,):

    current = nested_dict
    for key in keys_list:
        try:
            current = current[int(key)]
        except:
            current = current[key]
    return current

@dataclass
class FLLogger:
    """
    A simple logger for Federated Learning experiments.
    """

    num_clients: int
    output_dir: Path
    use_tensorboard: bool = False

    curr_round: list = field(default_factory=list)
    server_accs: list = field(default_factory=list)
    server_baccs: list = field(default_factory=list)
    server_f1wtd: list = field(default_factory=list)
    server_f1mic: list = field(default_factory=list)
    server_test_losses: list = field(default_factory=list)
    # server_ww_details: list = field(default_factory=list)
    # server_ww_summary: list = field(default_factory=list)
    lr_summary: list = field(default_factory=list)

    # agg_weights_log:    dict = field(default_factory=dict)
    client_trn_losses:  dict[int, list] = field(default_factory=dict)
    client_trn_accs:    dict[int, list] = field(default_factory=dict)
    client_trn_baccs:   dict[int, list] = field(default_factory=dict)
    client_test_accs:   dict[int, list] = field(default_factory=dict)
    client_test_baccs:  dict[int, list] = field(default_factory=dict)
    client_test_f1wtd:  dict[int, list] = field(default_factory=dict)
    client_test_f1mic:  dict[int, list] = field(default_factory=dict)
    client_test_losses: dict[int, list] = field(default_factory=dict)
    # client_ww_details:  dict = field(default_factory=dict)
    # client_ww_summary:  dict = field(default_factory=dict)
    client_ids : list = field(default_factory=list)

    def __post_init__(self):
        if self.client_ids == []:
            self.client_ids = [i for i in range(self.num_clients)]

        self.client_trn_losses =   {i: [] for i in self.client_ids}
        self.client_trn_accs =     {i: [] for i in self.client_ids}
        self.client_trn_baccs =    {i: [] for i in self.client_ids}
        self.client_test_accs =    {i: [] for i in self.client_ids}
        self.client_test_baccs =   {i: [] for i in self.client_ids}
        self.client_test_losses =  {i: [] for i in self.client_ids}
        self.client_test_f1wtd =   {i: [] for i in self.client_ids}
        self.client_test_f1mic =   {i: [] for i in self.client_ids}

        self.log_keys: list[str]= list(pd.json_normalize(asdict(self), sep='/').keys())
        self.log_keys.remove("output_dir")
        self.log_keys.remove("num_clients")
        self.log_keys.remove("use_tensorboard")
        self.scalar_log_keys = [key for key in self.log_keys if any(
            k in key for k in ["_accs", "_baccs", "_losses", "lr_", "rounds", "_sclr"])
        ]

        self.log_keys_steps = {key: 0 for key in self.log_keys}

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=Path("runs/")/self.output_dir.parent.name / self.output_dir.name)

    def resume_logger(self, outpath: Path):
        # FIXME: Currently only supports resuming non-dict log items
        self.use_tensorboard = False
        for key in self.log_keys:
            k1, *k2 = key.split('/')
            print(f"Resuming key: {k1}")
            if k1=='client_ids':
                continue
            if k1=='client_ww_details' or k1=='client_ww_summary' or k1=='server_ww_details' or k1=='server_ww_summary':
                continue
            # if 'client' in k1 and 'losses' in k1:
            if 'client' in k1:
                for i in self.client_ids:
                    if (outpath / f"{k1}_{i}.npy").exists():
                        key_array = np.load(outpath / f"{k1}_{i}.npy", allow_pickle=True)
                        value = key_array.tolist()
                        if hasattr(self, k1):
                            getattr(self, k1)[i] = value
                    else:
                        raise FileNotFoundError(f"File {outpath / f'{k1}_{i}.npy'} not found. Cannot resume logger.")
            # elif 'client'in k1:
            #     if (outpath / f"{k1}.npy").exists():
            #         key_array = np.load(outpath / f"{k1}.npy", allow_pickle=True)
            #         for i in self.client_ids:
            #             value = key_array[i].tolist()
            #             if hasattr(self, k1):
            #                 getattr(self, k1)[i] = value
            #     else:
            #         raise FileNotFoundError(f"File {outpath / f'{k1}.npy'} not found. Cannot resume logger.")
            else:
                if (outpath / f"{k1}.npy").exists():
                    key_array = np.load(outpath / f"{k1}.npy", allow_pickle=True)
                    value = key_array.tolist()
                    if hasattr(self, k1):
                        setattr(self, k1, value)
                else:
                    raise FileNotFoundError(f"File {outpath / f'{k1}.npy'} not found. Cannot resume logger.")

    def write_to_tensorboard(self):
        if not self.use_tensorboard:
            return

        for key in self.scalar_log_keys:
            k1, *k2 = key.split('/')
            val = getattr(self, k1, None)
            if isinstance(val, dict):
                # If the value is a dict, we need to fetch the nested value
                if k2:
                    val = get_nested_value(val, k2)
                else:
                    raise ValueError(f"Key {key} is a dict but no nested key provided.")

            assert isinstance(val, list) or isinstance(val, np.ndarray), \
                f"Value for key {key} is not a list or numpy array. Found: {type(val)}"
            

            # recursively fetch the value
            # for i, v in enumerate(val):
            if len(val) == 0:
                # print(f"Key {key} has no values to log.")
                continue
            elif len(val) == 1:
                # If there's only one value, log it directly
                self.writer.add_scalar(f"{key}", val[0], self.log_keys_steps[key])
                self.log_keys_steps[key] += 1
            else:
                # may need to revisit this logic if we want to flush the list time to time
                last_step = self.log_keys_steps[key]
                for j in range(last_step, len(val)):
                    self.writer.add_scalar(f"{key}", val[j], j)
                    self.log_keys_steps[key] += 1

    @staticmethod
    def get_final_value(log_list: list):
        if len(log_list) == 0:
            return np.nan
        return log_list[-1]
    
    
    def generate_summary(self):

        
        run_summary = {
                "global_model_test_accuracy": np.round(self.server_accs[-1], 5),
                "global_model_test_balanced_accuracy": np.round(self.server_baccs[-1], 5),
                "global_model_test_losses": np.round(self.server_test_losses[-1], 6),
                "client_train_losses_mean": np.mean(
                    [self.client_trn_losses[i][-1] for i in self.client_ids]
                ).round(6),
                "client_train_losses_std": np.std(
                    [self.client_trn_losses[i][-1] for i in self.client_ids]
                ).round(6),
                "client_train_accuracy_mean": np.mean(
                    [self.client_trn_accs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_train_accuracy_std": np.std(
                    [self.client_trn_accs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_train_balanced_accuracy_mean": np.mean(
                    [self.client_trn_baccs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_train_balanced_accuracy_std": np.std(
                    [self.client_trn_baccs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_test_accuracy_mean": np.mean(
                    [self.client_test_accs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_test_accuracy_std": np.std(
                    [self.client_test_accs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_test_balanced_accuracy_mean": np.mean(
                    [self.client_test_baccs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_test_balanced_accuracy_std": np.std(
                    [self.client_test_baccs[i][-1] for i in self.client_ids]
                ).round(5),
                "client_test_losses_mean": np.mean(
                    [self.client_test_losses[i][-1] for i in self.client_ids]
                ).round(6),
                "client_test_losses_std": np.std(
                    [self.client_test_losses[i][-1] for i in self.client_ids]
                ).round(6),
                "curr_round": self.curr_round[-1] + 1,
            }

        run_summary = pd.json_normalize(run_summary).to_dict(orient="records")[0]
        return run_summary
    
    def flush(self):
        """
        Flush the logs to the output directory.
        """
        # Save the logs to a file or database
        # This is a placeholder for actual implementation
        for key in self.log_keys:
            k1, *k2 = key.split('/')
            data = getattr(self, k1, None)

            if isinstance(data, list):
                array = np.array(data)
                np.save(self.output_dir / f"{k1}.npy", array)

            elif isinstance(data, dict):
                # print(f"Logging dictionary for key: {key}")
                # print(len(list(data.values())))
                if '_losses' in k1:
                    # If the data is a dict, we need to save each client's data separately
                    for client_id, values in data.items():
                        if len(values) == 0:
                            continue
                        np.save(self.output_dir / f"{k1}_{client_id}.npy", np.array(values))
                else:
                    # print(f"Logging dictionary for key: {k1}")
                    # print(data.values())
                    # try:
                    #     data_ar = np.array(list(data.values()))
                    #     np.save(self.output_dir / f"{k1}.npy", data_ar)
                    # except:
                    for client_id, values in data.items():
                        if len(values) == 0:
                            continue
                        np.save(self.output_dir / f"{k1}_{client_id}.npy", np.array(values))

            elif isinstance(data, np.ndarray):
                np.save(self.output_dir / f"{k1}.npy", data)
            elif isinstance(data, (int, float, str)):
                with open(self.output_dir / f"stray_variables.txt", "a") as f:
                    f.write(f'{k1}:{data}\n')
            else:
                raise TypeError(f"Unsupported type for logging: {type(data)}")
                    # with open(self.output_dir / f"{key}.json", "w") as f:
                    #     json.dump(data, f, indent=4)


# Checkpointing functions


def save_checkpoint(
    ckpt_dict: dict,
    actor: str,
    latest=True,
    suffix="",
    root_dir=".",
):

    if not os.path.exists(os.path.join(root_dir, "ckpts")):
        os.makedirs(os.path.join(root_dir, "ckpts"))

    if latest:
        fname = f"{actor}_latest.pt"
        fpath = os.path.join(root_dir, "ckpts", fname)
        if os.path.exists(fpath):
            os.rename(fpath, os.path.join(root_dir, "ckpts", f"{actor}_previous.pt"))
    else:
        fname = f"{actor}_{suffix}.pt"

    torch.save(ckpt_dict, os.path.join(root_dir, "ckpts", fname))


def load_checkpoint(
    actor: str,
    # model: Module,
    latest: bool = True,
    # optimizer: Optimizer = None,  # type: ignore
    # lrscheduler: LRScheduler = None,  # type: ignore
    root_dir=".",
    suffix: str = "",
) -> dict:

    ckpt_root = os.path.join(root_dir, "ckpts")
    if latest:
        ckpt_path = os.path.join(ckpt_root, f"{actor}_latest.pt")
    else:
        assert suffix != "", "Suffix must be provided for non-latest checkpoints"
        ckpt_path = os.path.join(ckpt_root, f"{actor}_{suffix}.pt")
    if not os.path.exists(ckpt_path):
        logging.warning(f"Checkpoint {ckpt_path} not found. Exiting.")
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found. Exiting.")

    logging.info(f"Loading actor {actor} checkpoint {ckpt_path}")

    ckpt = torch.load(ckpt_path)
    return ckpt


# Device and GPU management functions
def get_free_gpus(min_memory_reqd=4096):
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode()), names=["memory.used", "memory.free"], skiprows=1
    )
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    # min_memory_reqd = 10000
    ids = gpu_df.index[gpu_df["memory.free"] > min_memory_reqd]
    for id in ids:
        logging.debug(
            "Returning GPU:{} with {} free MiB".format(
                id, gpu_df.iloc[id]["memory.free"]
            )
        )
    return ids.to_list()


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode()), names=["memory.used", "memory.free"], skiprows=1
    )
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    # Choose the GPU with the most free memory except for gpu 7
    gpu_df = gpu_df.drop(7, axis=0, errors="ignore")  # Drop GPU 7 if it exists
    # print("GPU usage:\n{}".format(gpu_df))
    idx = gpu_df["memory.free"].idxmax()

    logging.debug("Returning GPU:{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"]))  # type: ignore
    return idx


def auto_configure_device():

    if torch.cuda.is_available():
        # Set visible GPUs
        # TODO: MAke the gpu configurable
        # gpu_ids = get_free_gpus()
        # logging.info('Selected GPUs:')
        # logging.info("Selected GPUs:" + ",".join(map(str, gpu_ids)))

        # Disabling below line due to cluster policies
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        if torch.cuda.device_count() > 1:
            if os.uname().nodename == "gigabyte-W771-Z00-00":
                device = "cuda:0"
            # elif os.uname().nodename == "srv-02":
            #     device = "cuda:5"
            else:
                device = f"cuda:{get_free_gpu()}"
        else:
            device = "cuda"

    else:
        device = "cpu"
    logging.info(f"Auto Configured device to: {device}")
    return device
