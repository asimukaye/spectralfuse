from copy import deepcopy
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchvision.models import ResNet
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score

## Root level module. Should not have any dependencies on other modules

#### Checkpointing functions

def adapt_model_last_layer(model: nn.Module, num_classes: int):
    """Adapt a ResNet model to a  number of classes.
    Args:
        model (nn.Module): Pretrained ResNet model
        num_classes (int): Number of output classes
    Returns:
        nn.Module: Adapted ResNet model"""
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes) # type: ignore
    elif hasattr(model, "classifier"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes) # type: ignore
    else:
        raise ValueError("Model does not have a fc or classifier layer.")
    return model

def get_accuracy(outputs: Tensor, targets: Tensor):
    """Calculate accuracy from outputs and targets.
    Args:
        outputs (Tensor): model outputs
        targets (Tensor): target labels
    Returns:
        accuracy (float): accuracy value"""
    preds = outputs.argmax(dim=1)
    acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
    return acc


# Train function
def train_one_epoch_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    losses = []
    model.to(device)
    predicted_all = torch.tensor([], dtype=torch.long)
    labels_all = torch.tensor([], dtype=torch.long)
    # train_loader
    model.train()
    bid = 0
    for batch in train_loader:
        bid += 1
        if isinstance(batch, dict):
            images, labels = batch['image'], batch['label']
        else:
            images, labels = batch
        labels_all = torch.cat((labels_all, labels.cpu()), dim=0)
        # print(images.shape)
        # print(labels.shape)
        # dump images and labels for debugging
        
        # torch.save(images, f"debug_images_batch_{bid}.pt")
        # torch.save(labels, f"debug_labels_batch_{bid}.pt")
        # exit(0)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        cls_outputs = model(images)
        outputs = cls_outputs.logits if hasattr(cls_outputs, "logits") else cls_outputs
        # print(outputs.shape) 

        loss = criterion(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        predicted_all = torch.cat((predicted_all, predicted.cpu()), dim=0)


    balanced_accuracy = balanced_accuracy_score(
        labels_all.numpy(), predicted_all.numpy()
    )
    accuracy = accuracy_score(labels_all.numpy(), predicted_all.numpy())
    return model, losses, accuracy, balanced_accuracy


def evaluate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    losses = []

    model.to(device)
    model.eval()
    # correct, total = 0, 0
    predicted_all = torch.tensor([], dtype=torch.long)
    labels_all = torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch
            labels_all = torch.cat((labels_all, labels.cpu()), dim=0)
            images, labels = images.to(device), labels.to(device)
            # outputs = model(images)
            cls_outputs = model(images)
            outputs = (
                cls_outputs.logits if hasattr(cls_outputs, "logits") else cls_outputs
            )
            _, predicted = torch.max(outputs, 1)
            predicted_all = torch.cat((predicted_all, predicted.cpu()), dim=0)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

    balanced_accuracy = balanced_accuracy_score(
        labels_all.numpy(), predicted_all.numpy()
    )
    accuracy = accuracy_score(labels_all.numpy(), predicted_all.numpy())
    f1_wtd = f1_score(labels_all.numpy(), predicted_all.numpy(), average='weighted')
    f1_micro = f1_score(labels_all.numpy(), predicted_all.numpy(), average='micro')

    return accuracy, balanced_accuracy, f1_wtd, f1_micro, losses


# Train function
def train_model(model, train_loader, epochs=1, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    losses = []
    model.to(device)
    # train_loader
    model.train()
    for e in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        # print(f"Epoch: {e} Loss: {loss.item()}")
    # model.to("cpu")
    # print("Training complete.")
    return model, losses



def train_single_clients(model_template, client_train_val_indices, criterion, num_clients,  epochs):
    # Training learning setup
    clients = {}
    # train_loaders, val_loaders = {}, {}
    val_evals = []
    # losses = {}
    train_losses = {}
    val_losses = {}
    for i, c_idx in client_train_val_indices.items():
        train_loader = DataLoader(
            Subset(dataset, c_idx[0]), batch_size=BATCH_SIZE, shuffle=True  # type: ignore
        )
        val_loader = DataLoader(
            Subset(dataset, c_idx[1]), batch_size=BATCH_SIZE, shuffle=False  # type: ignore
        )

        # model = SimpleCNN()
        model = deepcopy(model_template)
        model, loss = train_model(model, train_loader, epochs=epochs)
        val_eval, val_beval, val_f1w, val_f1micro,  val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Client {i} - Validation Accuracy: {val_eval}")
        train_losses[i] = loss
        val_evals.append(val_eval)
        val_losses[i] = val_loss
        clients[i] = model
        # train_loaders[i], val_loaders[i] = train_loader, val_loader

    test_evals = []
    # for i in range(num_clients):
    #     test_eval, test_beval, test_loss = evaluate_model(clients[i], test_loader, criterion)
    #     print(f"Client {i} - Test Accuracy: {test_eval}")
    #     test_evals.append(test_eval)

    # test_evals = np.array(test_evals)
    return clients, test_evals, val_evals, train_losses, val_losses

# Evaluation function
# def evaluate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module):
#     losses = []

#     model.to(DEVICE)
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
#             losses.append(loss.item())
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return correct / total, losses


# # Train function
# def train_one_epoch_model(
#     model: nn.Module,
#     train_loader: DataLoader,
#     optimizer: optim.Optimizer,
#     criterion: nn.Module,
# ):
#     losses = []
#     model.to(DEVICE)
#     # train_loader
#     model.train()
#     for images, labels in train_loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         losses.append(loss.item())
#         loss.backward()
#         optimizer.step()

#     return model, losses
