import os

import torch
import numpy as np

from model.model_factory import get_contrastive_modules, get_residual_models
from dataset.dataset import get_datasets
from train.train import *
from config.check_and_save import check_config_exist, add_entry, filter


def main(args):
    # filter invalid args
    if args.filter and filter(args):
        return
    
    # create output directory
    if args.save_checkpoint:
        os.makedirs(args.save_checkpoint_dir, exist_ok=True)
    if args.save_result or args.output_redirect:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.result_dir, exist_ok=True)
    
    # check if the setting already exists
    if check_config_exist(args) and not args.duplicate_enable and args.save_result:
        print("Setting already exists.")
        return
    
    print("Running on", args.device)
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # get modules
    feature_extractors, projection_heads, optimizer_feature, optimizer_projection = get_contrastive_modules(args)
    unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads, \
        unimodal_optimizers, bimodal_optimizers, trimodal_optimizers = get_residual_models(args)
        
    # get train, validation, and test datasets
    datasets = get_datasets(args)
    
    # train contrastive
    if not args.train_prediction_head_only:
        min_loss_contrastive = train_contrastive(
            datasets['training'], datasets['validation'],
            feature_extractors, projection_heads,
            optimizer_feature, optimizer_projection, args,
        )
    
    # train residual
    if args.header_training_paradigm == 'routing':
        min_loss_residual, best_metric_residual = train_residual_routing(
            datasets['training'], datasets['validation'], 
            feature_extractors, unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads,
            optimizer_feature, unimodal_optimizers, bimodal_optimizers, trimodal_optimizers,
            args,
        )
    else:
        min_loss_residual, best_metric_residual = train_residual_simultaneous(
            datasets['training'], datasets['validation'], 
            feature_extractors, unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads,
            optimizer_feature, unimodal_optimizers, bimodal_optimizers, trimodal_optimizers,
            args,
        )
    
    # test
    test_loss, test_metric_result = test(
        datasets['test'], 
        feature_extractors, unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads,
        args,
    )
    
    if args.save_checkpoint:
        # save checkpoint
        checkpoint = {
            'feature_extractors': {mod: model.state_dict() for mod, model in feature_extractors.items()},
            'projection_heads': {mod: model.state_dict() for mod, model in projection_heads.items()},
            'unimodal_prediction_heads': {mod: model.state_dict() for mod, model in unimodal_prediction_heads.items()},
            'bimodal_prediction_heads': {mod: model.state_dict() for mod, model in bimodal_prediction_heads.items()},
            'trimodal_prediction_heads': {mod: model.state_dict() for mod, model in trimodal_prediction_heads.items()},
        }
        torch.save(checkpoint, args.save_checkpoint_filepath)
    
    if args.save_result:
        # add an entry of training setting and results
        add_entry(args, test_metric_result)
