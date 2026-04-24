# spectralfuse
Code for running baseline federated averaging and the spectral aggregation variants used in the repository:

- `fedavg.py`: standard FedAvg / FedOpt style aggregation
- `spectralfed.py`: spectral-entropy weighting with Shapley-style contribution scores
- `spectralfuse.py`: the Kalman-fused spectral variant

The examples below are aimed at getting a basic run working on standard vision datasets such as CIFAR-10, CIFAR-100, MNIST, FashionMNIST, and EMNIST.

**Repository Layout**
- `fedavg.py`: entry point for FedAvg and FedOpt baselines
- `spectralfed.py`: entry point for spectral weighting experiments
- `spectralfuse.py`: entry point for the fused spectral weighting variant
- `data.py`: dataset loading and federated split construction
- `models.py`: model definitions used by the training scripts
- `output/`: created automatically for run artifacts
- `data/`: created automatically for downloaded datasets

**Environment Setup**
Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `requirements.txt` is enough for the default CIFAR/MNIST style runs documented here.
- If you want to use ViT-based models, install the optional `transformers` dependency.
- If you want to use `femnist` or `fedisic`, also install the optional `datasets`, `flwr-datasets`, and `flamby` dependencies.
- The first run will download torchvision datasets into `./data`.

**Quick Start**
Run a short smoke test first:

```bash
python fedavg.py --strategy fedavg_uni --dataset cifar10 --model tf_cnn --split iid --suffix quickstart --debug
```

If that works, try the spectral method:

```bash
python spectralfed.py --strategy spectralfed --dataset cifar10 --model tf_cnn --split iid --mu 0.9 --reward none --suffix quickstart --debug
```

You can also run the fused variant:

```bash
python spectralfuse.py --strategy spectralfuse --dataset cifar10 --model tf_cnn --split iid --mu 0.9 --reward none --suffix quickstart --debug
```

`--debug` reduces the run length so you can validate the setup quickly.

**Common Commands**
FedAvg with uniform client weighting:

```bash
python fedavg.py --strategy fedavg_uni --dataset cifar10 --model tf_cnn --split iid --suffix fedavg_iid
```

FedAvg with data-size weighting:

```bash
python fedavg.py --strategy fedavg --dataset cifar10 --model tf_cnn --split dirichlet_0.1 --suffix fedavg_dirichlet
```

FedOpt-style update:

```bash
python fedavg.py --strategy fedopt --dataset cifar10 --model tf_cnn --split only_label_skew --suffix fedopt_label_skew
```

SpectralFed on a non-IID split:

```bash
python spectralfed.py --strategy spectralfed --dataset cifar10 --model tf_cnn --split dirichlet_0.1 --mu 0.9 --reward none --suffix spectralfed_dirichlet
```

SpectralFuse on a non-IID split:

```bash
python spectralfuse.py --strategy spectralfuse --dataset cifar10 --model tf_cnn --split only_label_skew --mu 0.9 --reward none --suffix spectralfuse_label_skew
```

**Key Arguments**
- `--strategy`: aggregation strategy to run
  - `fedavg.py`: `fedavg`, `fedavg_uni`, `fedopt`, `fedopt_uni`
  - `spectralfed.py`: typically `spectralfed`
  - `spectralfuse.py`: typically `spectralfuse`
- `--dataset`: `cifar10`, `cifar100`, `mnist`, `fashionmnist`, `emnist`
- `--model`: `tf_cnn`, `mlpnet`, `resnet18`, `resnet34`, `resnet50`
- `--split`:
  - `iid`
  - `dirichlet_0.1` or another `dirichlet_<alpha>` value
  - `step_quantity`
  - `step_label_skew`
  - `only_label_skew`
- `--mu`: momentum/smoothing term used by the spectral methods
- `--reward`: reward mechanism for spectral methods, usually `none`
- `--suffix`: appended to the output run name
- `--resume_from`: resume a saved `fedavg.py` or `spectralfuse.py` run directory

**Outputs**
Each run creates a directory under:

```text
output/<strategy>-<dataset>-<model>/<timestamp>-<split>-<suffix>/
```

Typical files include:
- `config.yaml`: resolved configuration for the run
- `optimizer_cfg.json`
- `scheduler_cfg.json` when a scheduler is enabled
- `global_model.pth`
- `model_<client_id>.pth`
- `*.npy` metric logs flushed by the logger

**Resume Training**
`fedavg.py` and `spectralfuse.py` can resume from an existing run directory:

```bash
python fedavg.py --resume_from output/fedavg-cifar10-tf_cnn/<run_dir>
python spectralfuse.py --resume_from output/spectralfuse-cifar10-tf_cnn/<run_dir>
```

**Known Limits**
- `femnist` and `fedisic` rely on optional dataset packages and may require local dataset preparation.
- ViT models require the optional `transformers` package and usually more memory.
- The default scripts are configured for experimentation, not polished CLI packaging, so the quickest path is to start with `cifar10 + tf_cnn + --debug`.
