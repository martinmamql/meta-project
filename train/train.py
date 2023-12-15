from typing import Literal, Dict, Tuple, Optional
import math
import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import *
from dataset.dataset import *
from train.loss import multi_modal_contrastive_loss
from train.utils import *
from train.train_inner import *
from train.valid_inner import *
from train.info import ALL_MODALITIES


def train_contrastive(
    train_dataset: MMIDataset,
    valid_dataset: MMIDataset,
    
    feature_extractors: Dict[str, Extractor],
    projection_heads: Dict[str, ProjectionHead],
    feature_optimizers: Dict[str, torch.optim.Optimizer],
    projection_optimizers: Dict[str, torch.optim.Optimizer],
    
    args,
) -> Dict[str, float]:
    print("[Training contrastive]")
    # Dataloader
    train_dataloader = DataLoader(train_dataset, args.batch_size_contrastive, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size_contrastive, shuffle=False)
    
    # Init
    all_modality_pairs = get_pair_from_list(train_dataset.all_modalities)
    min_loss_hist_dict = {tuple(mod_pair): float('inf') for mod_pair in all_modality_pairs}
    epoch_cnt = 0
    no_decrease_in_loss = 0
    
    # Prepare copy for best model
    feature_extractors_best = copy.deepcopy(feature_extractors)
    projection_heads_best = copy.deepcopy(projection_heads)
    
    while epoch_cnt < args.num_epochs_contrastive and no_decrease_in_loss < args.patience_contrastive:
        # train
        train_contrastive_inner(train_dataloader, feature_extractors, projection_heads,
                                feature_optimizers, projection_optimizers, args)
            
        # validation
        valid_loss = valid_contrastive_inner(valid_dataloader, feature_extractors, projection_heads, args)
    
        # update minimum validation loss
        decrease_flag = False
        for mod_pair in all_modality_pairs:
            if valid_loss[mod_pair] < min_loss_hist_dict[mod_pair]:
                min_loss_hist_dict[mod_pair] = valid_loss[mod_pair]
                decrease_flag = True
                
        # update best model parameters
        if decrease_flag:
            print("Decrease in validation loss")
            feature_extractors_best = copy.deepcopy(feature_extractors)
            projection_heads_best = copy.deepcopy(projection_heads)
            
        # early stopping
        no_decrease_in_loss = no_decrease_in_loss + args.num_train_per_validation_residual if not decrease_flag else 0
        
        # print logs
        epoch_cnt += args.num_train_per_validation_contrastive
        if epoch_cnt % args.log_freq_contrastive == 0:
            print(f"train contrastive epoch {epoch_cnt} - validation loss:")
            for k, v in min_loss_hist_dict.items():
                print(f'\t{k} - {v}')
    
    # store the best model params back
    feature_extractors = copy.deepcopy(feature_extractors_best)
    projection_heads = copy.deepcopy(projection_heads_best)
    return min_loss_hist_dict


def train_residual_routing(
    train_dataset: MMIDataset,
    valid_dataset: MMIDataset,
    
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Optional[Dict[str, UnimodalModel]],
    bimodal_prediction_heads: Optional[Dict[str, BimodalModel]],
    trimodal_prediction_heads: Optional[Dict[str, TrimodalModel]],
    
    feature_extractor_optimizers: Dict[str, torch.optim.Optimizer],
    unimodal_optimizers: Optional[Dict[str, torch.optim.Optimizer]],
    bimodal_optimizers: Optional[Dict[str, torch.optim.Optimizer]],
    trimodal_optimizers: Optional[Dict[str, torch.optim.Optimizer]],
    
    args,
) -> Tuple[float, Dict[str, float]]:
    print("[Training residual routing]")
    # Dataloader
    train_dataloader = DataLoader(train_dataset, args.batch_size_residual, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size_residual, shuffle=False)
    
    # Init
    min_loss = float('inf')
    best_metric_result = initialize_metrix_result(args.metric_list)
    
    # Prepare copy for best model
    feature_extractors_best = copy.deepcopy(feature_extractors)
    unimodal_prediction_heads_best = copy.deepcopy(unimodal_prediction_heads)
    bimodal_prediction_heads_best = copy.deepcopy(bimodal_prediction_heads)
    trimodal_prediction_heads_best = copy.deepcopy(trimodal_prediction_heads)
    
    # Routing for each type of prediction heads
    
    # Unimodal
    print("[Training unimodal]")
    if args.header_type in ['residual', 'unimodal', 'uni_and_bi']:
        epoch_cnt = 0
        no_decrease_in_loss = 0
        
        while epoch_cnt < args.num_epochs_residual and no_decrease_in_loss < args.patience_residual:
            # training
            train_unimodal_inner(train_dataloader, feature_extractors, unimodal_prediction_heads, 
                                 feature_extractor_optimizers, unimodal_optimizers, args)
            
            # validation
            valid_loss, metric_result = valid_unimodal_inner(valid_dataloader, feature_extractors, unimodal_prediction_heads, args)
            
            # update minimum validation loss and best model parameters
            decrease_flag = False
            if valid_loss < min_loss:
                print("Decrease in validation loss")
                decrease_flag = True
                min_loss = valid_loss
                
                if not args.freeze_contrastive_param:
                    feature_extractors_best = copy.deepcopy(feature_extractors)
                unimodal_prediction_heads_best = copy.deepcopy(unimodal_prediction_heads)
                
                updated = update_best_result(metric_result, best_metric_result)
                
            # early stopping
            no_decrease_in_loss = no_decrease_in_loss + args.num_train_per_validation_residual if not decrease_flag else 0
            
            # print logs
            epoch_cnt += args.num_train_per_validation_residual
            if epoch_cnt % args.log_freq_residual == 0:
                print(f"[train unimodal] epoch {epoch_cnt}\n- validation loss: {valid_loss}")
                print(f"- best metric result:{metric_result}")
                
        # store the best model params back
        if not args.freeze_contrastive_param:
            feature_extractors = copy.deepcopy(feature_extractors_best)
        unimodal_prediction_heads = copy.deepcopy(unimodal_prediction_heads_best)
        
    # Bimodal
    print("[Training bimodal]")
    if args.header_type in ['residual', 'uni_and_bi', 'bimodal']:
        
        # warmup
        #? necessary?
        epoch_cnt = 0
        while epoch_cnt < args.warmup_epochs:
            # training
            train_bimodal_inner(train_dataloader, feature_extractors, 
                                unimodal_prediction_heads, bimodal_prediction_heads, 
                                feature_extractor_optimizers, bimodal_optimizers, args)
            
            # validation
            valid_loss, metric_result = \
                valid_bimodal_inner(valid_dataloader, feature_extractors, 
                                    unimodal_prediction_heads, bimodal_prediction_heads, args)
            
            # print logs
            epoch_cnt += args.num_train_per_validation_residual
            print(f"[train bimodal warmup] epoch {epoch_cnt}\n- validation loss: {valid_loss}")
            print(f"- best metric result:{metric_result}")
        
        # training and validation
        epoch_cnt = 0
        no_decrease_in_loss = 0
        
        while epoch_cnt < args.num_epochs_residual and no_decrease_in_loss < args.patience_residual:
            # training
            train_bimodal_inner(train_dataloader, feature_extractors, 
                                unimodal_prediction_heads, bimodal_prediction_heads, 
                                feature_extractor_optimizers, bimodal_optimizers, args)
            
            # validation
            valid_loss, metric_result = \
                valid_bimodal_inner(valid_dataloader, feature_extractors, 
                                    unimodal_prediction_heads, bimodal_prediction_heads, args)
            
            # update minimum validation loss and best model parameters
            decrease_flag = False
            if valid_loss < min_loss:
                print("Decrease in validation loss")
                decrease_flag = True
                min_loss = valid_loss
                
                if not args.freeze_contrastive_param:
                    feature_extractors_best = copy.deepcopy(feature_extractors)
                bimodal_prediction_heads_best = copy.deepcopy(bimodal_prediction_heads)
                
                updated = update_best_result(metric_result, best_metric_result)
                
            # early stopping
            no_decrease_in_loss = no_decrease_in_loss + args.num_train_per_validation_residual if not decrease_flag else 0
            
            # print logs
            epoch_cnt += args.num_train_per_validation_residual
            if epoch_cnt % args.log_freq_residual == 0:
                print(f"[train bimodal] epoch {epoch_cnt}\n- validation loss: {valid_loss}")
                print(f"- best metric result:{metric_result}")
                
        # store the best model params back
        if not args.freeze_contrastive_param:
            feature_extractors = copy.deepcopy(feature_extractors_best)
        bimodal_prediction_heads = copy.deepcopy(bimodal_prediction_heads_best)
    
    # Trimodal
    print("[Training trimodal]")
    if args.header_type in ['residual', 'trimodal']:
        
        # warmup
        #? necessary?
        epoch_cnt = 0
        while epoch_cnt < args.warmup_epochs:
            # training
            train_trimodal_inner(train_dataloader, feature_extractors, 
                                 unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads, 
                                 feature_extractor_optimizers, trimodal_optimizers, args)
            
            # validation
            valid_loss, metric_result = \
                valid_trimodal_inner(valid_dataloader, feature_extractors, 
                                     unimodal_prediction_heads, bimodal_prediction_heads, 
                                     trimodal_prediction_heads, args)
            
            # print logs
            epoch_cnt += args.num_train_per_validation_residual
            print(f"[train trimodal warmup] epoch {epoch_cnt}\n- validation loss: {valid_loss}")
            print(f"- best metric result:{metric_result}")
        
        # training and validation
        epoch_cnt = 0
        no_decrease_in_loss = 0
        
        while epoch_cnt < args.num_epochs_residual and no_decrease_in_loss < args.patience_residual:
            # training
            train_trimodal_inner(train_dataloader, feature_extractors, 
                                 unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads, 
                                 feature_extractor_optimizers, trimodal_optimizers, args)
            
            # validation
            valid_loss, metric_result = \
                valid_trimodal_inner(valid_dataloader, feature_extractors, 
                                     unimodal_prediction_heads, bimodal_prediction_heads, 
                                     trimodal_prediction_heads, args)
            
            # update minimum validation loss and best model parameters
            decrease_flag = False
            if valid_loss < min_loss:
                print("Decrease in validation loss")
                decrease_flag = True
                min_loss = valid_loss
                
                if not args.freeze_contrastive_param:
                    feature_extractors_best = copy.deepcopy(feature_extractors)
                trimodal_prediction_heads_best = copy.deepcopy(trimodal_prediction_heads)
                
                updated = update_best_result(metric_result, best_metric_result)
                
            # early stopping
            no_decrease_in_loss = no_decrease_in_loss + args.num_train_per_validation_residual if not decrease_flag else 0
            
            # print logs
            epoch_cnt += args.num_train_per_validation_residual
            if epoch_cnt % args.log_freq_residual == 0:
                print(f"[train trimodal] epoch {epoch_cnt}\n- validation loss: {valid_loss}")
                print(f"- best metric result:{metric_result}")
                
        # store the best model params back
        if not args.freeze_contrastive_param:
            feature_extractors = copy.deepcopy(feature_extractors_best)
        trimodal_prediction_heads = copy.deepcopy(trimodal_prediction_heads_best)
        
    return min_loss, best_metric_result


def train_residual_simultaneous(
    train_dataset: MMIDataset,
    valid_dataset: MMIDataset,
    
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Optional[Dict[str, UnimodalModel]],
    bimodal_prediction_heads: Optional[Dict[str, BimodalModel]],
    trimodal_prediction_heads: Optional[Dict[str, TrimodalModel]],
    
    feature_extractor_optimizers: Dict[str, torch.optim.Optimizer],
    unimodal_optimizers: Optional[Dict[str, torch.optim.Optimizer]],
    bimodal_optimizers: Optional[Dict[str, torch.optim.Optimizer]],
    trimodal_optimizers: Optional[Dict[str, torch.optim.Optimizer]],
    
    args,
) -> Tuple[float, Dict[str, float]]:
    print("[Training residual simultaneous]")
    # Dataloader
    train_dataloader = DataLoader(train_dataset, args.batch_size_residual, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size_residual, shuffle=False)
    
    # Init
    epoch_cnt = 0
    no_decrease_in_loss = 0
    min_loss = float('inf')
    best_metric_result = initialize_metrix_result(args.metric_list)
    
    # Prepare copy for best model
    feature_extractors_best = copy.deepcopy(feature_extractors)
    unimodal_prediction_heads_best = copy.deepcopy(unimodal_prediction_heads)
    bimodal_prediction_heads_best = copy.deepcopy(bimodal_prediction_heads)
    trimodal_prediction_heads_best = copy.deepcopy(trimodal_prediction_heads)
    
    # TRAIN loop
    while no_decrease_in_loss < args.patience_residual:
        # training
        train_simultaneous_inner(train_dataloader, feature_extractors, 
                                 unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads, 
                                 feature_extractor_optimizers, unimodal_optimizers, 
                                 bimodal_optimizers, trimodal_optimizers, args)
        
        # validation
        valid_loss, metric_result = \
            valid_trimodal_inner(valid_dataloader, feature_extractors, 
                                 unimodal_prediction_heads, bimodal_prediction_heads, 
                                 trimodal_prediction_heads, args)
        
        # update minimum validation loss and best model parameters
        decrease_flag = False
        if valid_loss < min_loss:
            print("Decrease in validation loss")
            decrease_flag = True
            min_loss = valid_loss
            
            if not args.freeze_contrastive_param:
                feature_extractors_best = copy.deepcopy(feature_extractors)
            unimodal_prediction_heads_best = copy.deepcopy(unimodal_prediction_heads)
            bimodal_prediction_heads_best = copy.deepcopy(bimodal_prediction_heads)
            trimodal_prediction_heads_best = copy.deepcopy(trimodal_prediction_heads)
            
            updated = update_best_result(metric_result, best_metric_result)
            
        # early stopping
        no_decrease_in_loss = no_decrease_in_loss + args.num_train_per_validation_residual if not decrease_flag else 0
        
        # print logs
        epoch_cnt += args.num_train_per_validation_residual
        if epoch_cnt % args.log_freq_residual == 0:
            print(f"[train simultaneous] epoch {epoch_cnt}\n- validation loss: {valid_loss}")
            print(f"- best metric result:{metric_result}")
            
    # store the best model params back
    if not args.freeze_contrastive_param:
        feature_extractors = copy.deepcopy(feature_extractors_best)
    unimodal_prediction_heads = copy.deepcopy(unimodal_prediction_heads_best)
    bimodal_prediction_heads = copy.deepcopy(bimodal_prediction_heads_best)
    trimodal_prediction_heads = copy.deepcopy(trimodal_prediction_heads_best)
    
    return min_loss, best_metric_result


def test(
    test_dataset: MMIDataset,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Optional[Dict[str, UnimodalModel]],
    bimodal_prediction_heads: Optional[Dict[str, BimodalModel]],
    trimodal_prediction_heads: Optional[Dict[str, TrimodalModel]],
    args
) -> Tuple[float, Dict[str, float]]:
    # Dataloader
    test_dataloader = DataLoader(test_dataset, args.batch_size_test, shuffle=False)

    if args.header_type == 'unimodal':
        min_loss, best_metric_result = valid_unimodal_inner(test_dataloader, feature_extractors, unimodal_prediction_heads, args)
    
    elif args.header_type in ['uni_and_bi', 'bimodal']:
        min_loss, best_metric_result = valid_bimodal_inner(test_dataloader, feature_extractors, 
                                                           unimodal_prediction_heads, bimodal_prediction_heads, args)
        
    elif args.header_type in ['residual', 'trimodal']:
        min_loss, best_metric_result = valid_trimodal_inner(test_dataloader, feature_extractors, 
                                                            unimodal_prediction_heads, bimodal_prediction_heads, 
                                                            trimodal_prediction_heads, args)
    
    else:
        raise ValueError(f'Invalid header type: {args.header_type}')
    
    print(f"[test]\n- loss: {min_loss}\n- best metric result:{best_metric_result}")
    return min_loss, best_metric_result