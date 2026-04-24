from functools import partial
import torch
from torch import nn
from torch.nn import Module
from torch import Tensor
import logging
import torch.nn.functional as F
import inspect
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, alexnet

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x


class RFFL_CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(RFFL_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # ic(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # ic(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # ic(x.shape)
        x = x.view(x.size(0), -1)

        # x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNCifar_TF(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, device=None):
        super(CNNCifar_TF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(-1, 64 * 4 * 4)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

class FedNet(nn.Module):
    def __init__(self, in_channels, bias=False, num_classes=10):
        super(FedNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, bias=bias)
        self.fc1 = nn.Linear(64 * 5 * 5, 512, bias=bias)
        self.fc2 = nn.Linear(512, 128, bias=bias)
        self.fc3 = nn.Linear(128, num_classes, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def resnet50_pretrained(num_classes=10 ):
    return _build_torchvision_classifier(resnet50, num_classes=num_classes, pretrained=True)

def vit_b_16_pretrained(num_classes=10):
    from transformers.models.vit import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True
    )
    return model

def resnet18_wrapper(num_classes=10):
    return _build_torchvision_classifier(resnet18, num_classes=num_classes, pretrained=False)

def resnet34_wrapper(num_classes=10):
    return _build_torchvision_classifier(resnet34, num_classes=num_classes, pretrained=False)

def resnet50_wrapper(num_classes=10):
    return _build_torchvision_classifier(resnet50, num_classes=num_classes, pretrained=False)

class MLP_Net(nn.Module):

    def __init__(self, num_classes=10, in_features=1024):
        self.in_features = in_features
        super(MLP_Net, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)            # [batch, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)           # mean pooling
        out = self.fc(pooled)                   # [batch, num_classes]
        return out
    

CLF_MODEL_MAP = {
    "simplecnn": SimpleCNN,
    # "twocnn": TwoCNN,
    # "twocnnv2": TwoCNNv2,
    "mlpnet": MLP_Net,
    "fednet": FedNet,
    # "resnet18_custom": ResNet18,
    # "resnet18": partial(resnet_wrapper, model_name="resnet18"),
    "resnet18": resnet18_wrapper,
    "resnet34": resnet34_wrapper,
    "resnet50": resnet50_wrapper,
    "resnet50_pret": resnet50_pretrained,
    "vit_b_16_pret": vit_b_16_pretrained,
    # "resnet34_custom": ResNet34,
    "rffl_cnn": RFFL_CNN,
    "tf_cnn": CNNCifar_TF,
    # "mlpnet": MLP_Net,
}


def _build_torchvision_classifier(model_fn, num_classes: int, pretrained: bool):
    kwargs = {"num_classes": num_classes}
    signature = inspect.signature(model_fn)
    if "weights" in signature.parameters:
        if pretrained:
            weights_map = {
                "resnet18": getattr(torchvision.models, "ResNet18_Weights", None),
                "resnet34": getattr(torchvision.models, "ResNet34_Weights", None),
                "resnet50": getattr(torchvision.models, "ResNet50_Weights", None),
            }
            weight_enum = weights_map.get(model_fn.__name__)
            kwargs["weights"] = weight_enum.DEFAULT if weight_enum is not None else None
        else:
            kwargs["weights"] = None
    elif "pretrained" in signature.parameters:
        kwargs["pretrained"] = pretrained
    return model_fn(**kwargs)


#########################
# Weight initialization #
#########################
def init_weights(model: Module, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """

    def init_func(m: Module):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight"):
                if isinstance(m.weight, Tensor):
                    torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, "bias"):
                if isinstance(m.bias, Tensor):
                    torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Conv") != -1 or classname.find("Linear") != -1:
            if hasattr(m, "weight"):
                if isinstance(m.weight, Tensor):
                    if init_type == "normal":
                        torch.nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
                    elif init_type == "xavier":
                        torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                    elif init_type == "xavier_uniform":
                        torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                    elif init_type == "kaiming":
                        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                    elif init_type == "orthogonal":
                        torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                    elif init_type == "none":  # uses pytorch's default init method
                        m.reset_parameters()  # type: ignore
                    else:
                        raise NotImplementedError(
                            f"[ERROR] Initialization method {init_type} is not implemented!"
                        )
            if hasattr(m, "bias") and m.bias is not None:
                if isinstance(m.bias, Tensor):
                    torch.nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def get_clf_model(model_name, metadata: dict) -> Module:
    
    # initialize the model class
    model_func = CLF_MODEL_MAP[model_name]
    # init_weights(model, init_type, init_gain)
    # Check arguments of model_func
    all_args = inspect.signature(model_func).parameters
    kwargs = {}
    for name, param in all_args.items():
        if name in metadata:
            kwargs[name] = metadata[name]
        elif param.default != inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            raise ValueError(f"Missing required parameter '{name}' for model '{model_name}'")
    
    model = model_func(**kwargs)
    return model
