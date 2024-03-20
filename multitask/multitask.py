import os
import copy
import random

import torch
from torch.utils.data import Dataset
import numpy as np

from model.model_factory import get_contrastive_modules, get_residual_models
from train.train import *
from config.check_and_save import check_config_exist_multitask, add_entry_multitask, filter


def multitask(args):
    assert args.save_checkpoint
    
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
    if args.save_result and check_config_exist_multitask(args) and not args.duplicate_enable:
        print("Setting already exists.")
        return
    
    print("Running on", args.device)
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
    # get modules
    feature_extractors, projection_heads, optimizer_feature, optimizer_projection = get_contrastive_modules(args)
    
    # get datasets
    multitask_training_datasets, multitask_validation_datasets = get_multitask_datasets(args)
    
    # run
    for loop in range(args.loop_multitask_times):
        print(f"\n\n# LOOP No.{loop}")
        training_datasets = copy.deepcopy(multitask_training_datasets)
        
        while True:
            for dataset_name in args.dataset_name_list:
                if len(training_datasets[dataset_name]) == 0:
                    continue
                
                print(f"\n<Training on {dataset_name}>")
                train_dataset = training_datasets[dataset_name].pop()
                valid_dataset = multitask_validation_datasets[dataset_name]
                # train contrastive
                min_loss_contrastive = train_contrastive(
                    train_dataset, valid_dataset,
                    feature_extractors, projection_heads,
                    optimizer_feature, optimizer_projection, args,
                )
            
            # check if all datasets are empty
            if max([len(datasets) for datasets in training_datasets.values()]) == 0:
                break
    
    # save checkpoint
    checkpoint = {
        'feature_extractors': {mod: model.state_dict() for mod, model in feature_extractors.items()},
        'projection_heads': {mod: model.state_dict() for mod, model in projection_heads.items()},
    }
    torch.save(checkpoint, args.save_checkpoint_filepath)
    
    if args.save_result:
        # add an entry of training setting and results
        add_entry_multitask(args, min_loss_contrastive)