from copy import deepcopy
import math
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Iterator


def mask_grad_update_by_order(
    grad_update: list[Tensor], mask_order=None, mask_percentile=None, mode="all"
):

    if mode == "all":
        # mask all but the largest <mask_order> updates (by magnitude) to zero
        all_update_mod = torch.cat(
            [update.data.view(-1).abs() for update in grad_update]
        )
        if not mask_order and mask_percentile is not None:
            # D is used implicitly here as len(all_update_mod)
            mask_order = int(len(all_update_mod) * mask_percentile)
            # print(f"Masking {mask_order} updates by percentile {mask_percentile}")
    
        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float("inf"))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order)  # type: ignore
            return mask_grad_update_by_magnitude(grad_update, topk[-1])

    elif mode == "layer":  # layer wise   largest-values criterion
        grad_update = deepcopy(grad_update)

        mask_percentile = max(0, mask_percentile)  # type: ignore
        for i, layer in enumerate(grad_update):
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)

            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(
                    layer_mod, min(mask_order, len(layer_mod) - 1)  # type: ignore
                )
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update
    else:
        raise ValueError("Invalid grad update by order mode")


def mask_grad_update_by_magnitude(grad_update: list[Tensor], mask_constant):

    # mask all but the updates with larger magnitude than <mask_constant> to zero
    # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
    grad_update = deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update


def interpolation_rewards(global_params: Iterator[Parameter], client_params: Iterator[Parameter], coeff: Tensor):
    # coeff_tensor = torch.tensor(coeff, device=device)

    for global_param, client_param in zip(global_params, client_params):
        # personalization
        # Check all the data types
        # print(client_param.data.dtype, global_param.data.dtype, coeff.dtype)
        client_param.data = (1 - coeff) * client_param.data + coeff * global_param.data

def sparsification_gradient_rewards(client_params: Iterator[Parameter], agg_gradient,  coeff:Tensor,  beta = 1.0):

    q_ratio = torch.tanh(beta * coeff) / torch.max(torch.tanh(beta * coeff))

    reward_gradient = mask_grad_update_by_order(
                        agg_gradient, mask_percentile=q_ratio, mode="all"
                )
    
    for client_param, reward_grad in zip(client_params, reward_gradient):
        client_param.data.add_(reward_grad) 
    
def sparsification_param_rewards(client_params: Iterator[Parameter], global_params: Iterator[Parameter],  coeff:Tensor,  beta = 1.0):

    q_ratio = torch.tanh(beta * coeff) / torch.max(torch.tanh(beta * coeff))

    reward_params = mask_grad_update_by_order(list(global_params), mask_percentile=q_ratio, mode="all")
    for client_param, reward_param in zip(client_params, reward_params):
        client_param.data.copy_(reward_param)

def no_rewards(global_params: Iterator[Parameter], client_params: Iterator[Parameter]):
    for global_param, client_param in zip(global_params, client_params):
        client_param.data.copy_(global_param.data)
