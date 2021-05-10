import models
from torch.nn.utils import prune
import torch
import statistics


def prune_transformer_block(transformer_block, args):
    pruning_amount = float(args.pruning_amount)
    prune.ln_structured(transformer_block.fc1, name='weight', amount=pruning_amount, n=0, dim=0)
    prune.remove(transformer_block.fc1, 'weight')
    prune.ln_structured(transformer_block.fc2, name='weight', amount=pruning_amount, n=0, dim=0)
    prune.remove(transformer_block.fc2, 'weight')
    for sub_module in transformer_block.fc_delta:
        if isinstance(sub_module, torch.nn.Linear):
            prune.ln_structured(sub_module, name='weight', amount=pruning_amount, n=0, dim=0)
            prune.remove(sub_module, 'weight')
    for sub_module in transformer_block.fc_gamma:
        if isinstance(sub_module, torch.nn.Linear):
            prune.ln_structured(sub_module, name='weight', amount=pruning_amount, n=0, dim=0)
            prune.remove(sub_module, 'weight')
    return transformer_block


def prune_model(model, args):
    pruning_style = args.pruning_style
    prune_layers = []
    if pruning_style == 'bottom':
        prune_layers = [1, 2, 3]
    if pruning_style == ' alternate':
        prune_layers = [1, 3, 5]
    if pruning_style == 'top':
        prune_layers = [3, 4, 5]
    if pruning_style == 'mid':
        prune_layers = [2, 3, 4]
    if pruning_style == 'all':
        prune_layers = [1, 2, 3, 4, 5]
    transformer_block_count = 0
    for idx in range(len(list(model.modules()))):
        module = list(model.modules())[idx]
        if isinstance(module, models.Sumanu.transformer.TransformerBlock):
            transformer_block_count += 1
            if transformer_block_count in prune_layers:
                module = prune_transformer_block(module, args)
                list(model.modules())[idx] = module
    return model


def get_sparsity(layer):
    return 100. * float(torch.sum(layer.weight == 0)) / float(layer.weight.nelement())


def show_transformer_sparsity(model):
    sparsity_list = []
    for idx in range(len(list(model.modules()))):
        module = list(model.modules())[idx]
        if isinstance(module, models.Sumanu.transformer.TransformerBlock):
            sparsity_list.append(get_sparsity(module.fc1))
            sparsity_list.append(get_sparsity(module.fc2))
    print("Current model sparsity : {:.2f}%".format(statistics.mean(sparsity_list)))
