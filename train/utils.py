from typing import Dict, Tuple, List, Union, Optional

import torch
from torch import nn

from train.info import ALL_MODALITIES, ALL_MODALITY_PAIRS, ALL_MODALITY_TRIPLETS, ALL_METRICES, ALL_LOSS
from train.loss import *
from train.metrics import *

EPS = 1e-6


def get_pair_from_list(modality_names: list) -> List[Tuple[str, str]]:
    modality_names = list(modality_names)
    assert all(mod in ALL_MODALITIES for mod in modality_names)
    
    pairs = []
    for i in range(len(modality_names)):
        for j in range(len(modality_names)):
            if [modality_names[i], modality_names[j]] in ALL_MODALITY_PAIRS:
                pairs.append((modality_names[i], modality_names[j]))
    return pairs


def get_triplet_from_list(modality_names: list) -> List[Tuple[str, str]]:
    modality_names = list(modality_names)
    assert all(mod in ALL_MODALITIES for mod in modality_names)
    
    triplets = []
    for i in range(len(modality_names)):
        for j in range(len(modality_names)):
            for k in range(len(modality_names)):
                if [modality_names[i], modality_names[j], modality_names[k]] in ALL_MODALITY_TRIPLETS:
                    triplets.append((modality_names[i], modality_names[j], modality_names[k]))
    return triplets


def loss_and_metric_for_dataset(dataset_name: str) -> Tuple[str, List[str]]:
    if dataset_name in ['mosi_sentiment', 'mosei_happiness', 'mosei_sentiment']:
        return 'mae+corr', ['mae', 'corr']
    elif dataset_name in ['vreed_av']:
        return 'ce', ['acc', 'wf1']
    elif dataset_name in ['sewa_arousal', 'sewa_valence']:
        return 'mae+ccc', ['mae', 'ccc']
    elif dataset_name in ['recola_arousal', 'recola_valence']:
        return 'mae+ccc', ['mae', 'ccc']
    elif dataset_name in ['iemocap_valence', 'iemocap_arousal']:
        return 'mae+ccc', ['mae', 'ccc']
    else:
        raise ValueError(f"Wrong dataset name: {dataset_name}")


def get_loss_function(loss_fn_name: str, alpha: float = 0.9):
    assert loss_fn_name in ALL_LOSS
    if loss_fn_name == 'ccc':
        return ccc_loss
    elif loss_fn_name == 'corr':
        return correlation_loss
    elif loss_fn_name == 'ce':
        return cross_entropy_loss
    elif loss_fn_name == 'mae':
        return nn.L1Loss()
    elif loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'wf1':
        return f1_loss
    elif loss_fn_name == 'acc':
        return accuracy_loss
    elif loss_fn_name == 'mae+corr':
        return lambda x, y: alpha * nn.L1Loss()(x, y) + (1 - alpha) * correlation_loss(x, y)
    elif loss_fn_name == 'mae+corr+ce':
        return lambda x, y: alpha * nn.L1Loss()(x, y) + (1 - alpha) * correlation_loss(x, y) + cross_entropy_loss(x, y)
    elif loss_fn_name == 'mae+ccc':
        return lambda x, y: alpha * nn.L1Loss()(x, y) + (1 - alpha) * ccc_loss(x, y)
    elif loss_fn_name == 'mae+corr+ccc':
        return lambda x, y: alpha * nn.L1Loss()(x, y) + 0.5 * (1 - alpha) * correlation_loss(x, y) + 0.5 * (1 - alpha) * ccc_loss(x, y)

def get_metrix_function(metrices: List[str]) -> dict:
    assert all(metrix in ALL_METRICES for metrix in metrices)
    
    function_dict = {}
    for m in metrices:
        if m == 'ccc':
            function_dict[m] = concordance_correlation_coefficient
        elif m == 'corr':
            function_dict[m] = correlation
        elif m == 'mae':
            function_dict[m] = mean_absolute_error
        elif m == 'mse':
            function_dict[m] = mean_squared_error
        elif m == 'wf1':
            function_dict[m] = weighted_one_vs_rest_f1
        elif m == 'acc':
            function_dict[m] = accuracy
    
    return function_dict


def get_result_dict(metrices: List[str]) -> dict:
    assert all(metrix in ALL_METRICES for metrix in metrices)
    
    result_dict = {}
    for m in metrices:
        if m == 'ccc':
            result_dict[m] = 0.0
        elif m == 'corr':
            result_dict[m] = 0.0
        elif m == 'mae':
            result_dict[m] = 0.0
        elif m == 'mse':
            result_dict[m] = 0.0
        elif m == 'wf1':
            result_dict['f'] = 0.0
            result_dict['p'] = 0.0
            result_dict['r'] = 0.0
        elif m == 'acc':
            result_dict[m] = 0.0
            
    return result_dict


def zero_init_metric_result(metrices: List[str]):
    assert all(metric in ALL_METRICES for metric in metrices)
    result_dict = {}
    for m in metrices:
        if m == 'ccc':
            result_dict[m] = 0.0
        elif m == 'corr':
            result_dict[m] = 0.0
        elif m == 'mae':
            result_dict[m] = 0.0
        elif m == 'mse':
            result_dict[m] = 0.0
        elif m == 'wf1':
            result_dict['f'] = 0.0
            result_dict['p'] = 0.0
            result_dict['r'] = 0.0
        elif m == 'acc':
            result_dict[m] = 0.0
            
    return result_dict


def initialize_metrix_result(metrices: List[str]):
    assert all(metric in ALL_METRICES for metric in metrices)
    
    result_dict = {}
    for m in metrices:
        if m == 'ccc':
            result_dict[m] = 0.0
        elif m == 'corr':
            result_dict[m] = 0.0
        elif m == 'mae':
            result_dict[m] = float('inf')
        elif m == 'mse':
            result_dict[m] = float('inf')
        elif m == 'wf1':
            result_dict['f'] = 0.0
            result_dict['p'] = 0.0
            result_dict['r'] = 0.0
        elif m == 'acc':
            result_dict[m] = 0.0
            
    return result_dict


def update_best_result(result_dict: Optional[Dict[str, float]], best_result_dict: Dict[str, float]) -> bool:
    updated = False
    
    if result_dict is None:
        return updated
    for k, v in result_dict.items():
        if k in ['mae', 'mse']:
            if best_result_dict[k] > result_dict[k]:
                best_result_dict[k] = result_dict[k]
                updated = True
        else:
            if best_result_dict[k] < result_dict[k]:
                best_result_dict[k] = result_dict[k]
                updated = True
    
    return updated


def update_result(res: Union[torch.tensor, List[torch.tensor]], metrics: str, result_dict: dict, label_std: Optional[float] = None) -> None:
    assert metrics in ALL_METRICES
    
    if metrics == 'mae':
        result_dict[metrics] += res.item() * label_std
    elif metrics != 'wf1':
        result_dict[metrics] += res.item()
    else:
        result_dict['f'] += res[0].item()
        result_dict['p'] += res[0].item()
        result_dict['r'] += res[0].item()

