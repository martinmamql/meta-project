import os
from typing import List, Dict

import pandas as pd

INDEX_COLUMNS = [
    'dataset_name', # Data
    'contrastive', 'header_type', 'header_training_paradigm', # Training Setting
    'in_batch_cl', 'freeze_contrastive_param', 'multitask',
    'batch_size_contrastive', 'batch_size_residual', # Hyperparameters
    'lr_contrastive', 'mm_contrastive', 'wd_contrastive', 'optim_contrastive',
    'feature_input_dim', 'feature_hidden_dim', 'feature_output_dim', 'feature_extractor_depth',
    'projection_output_dim', 
    'lr_unimodal', 'lr_bimodal', 'lr_trimodal',
    'mm_unimodal', 'mm_bimodal', 'mm_trimodal',
    'wd_unimodal', 'wd_bimodal', 'wd_trimodal',
    'optim_unimodal', 'optim_bimodal', 'optim_trimodal',
    'prediction_head_depth',
    'num_epochs_contrastive', 'num_epochs_residual',  # Epoch and patience
    'patience_contrastive', 'patience_residual', 'warmup_epochs',
    'load_pretrain_contrastive', 'load_pretrain_residual', # Checkpoint
    'load_checkpoint_filepath', 
    'loss_fn_name', 'metric_list', 'seed', # Misc
]
ALL_COLUMNS = INDEX_COLUMNS + ['save_checkpoint_filepath', 'best_metric_result']

INDEX_COLUMNS_MULTITASK = [
    'contrastive', # Training Setting
    'in_batch_cl', 'batch_size_contrastive', # Hyperparameters
    'lr_contrastive', 'mm_contrastive', 'wd_contrastive', 'optim_contrastive',
    'feature_input_dim', 'feature_hidden_dim', 'feature_output_dim', 'feature_extractor_depth',
    'projection_output_dim',
    'num_epochs_contrastive', 'patience_contrastive', # Epoch and patience
    'seed', # Misc
    'loop_multitask_times', # Multitask
    'dataset_name_list', 'per_dataset_size', 'balanced'
]
ALL_COLUMNS_MULTITASK = INDEX_COLUMNS_MULTITASK + ['save_checkpoint_filepath', 'loss_contrastive']


def get_args_info(args) -> List[str]:
    args_info = [
        args.dataset_name, # Data
        args.contrastive, args.header_type, args.header_training_paradigm, # Training Setting
        args.in_batch_cl, args.freeze_contrastive_param, args.multitask,
        args.batch_size_contrastive, args.batch_size_residual, # Hyperparameters
        args.lr_contrastive, args.mm_contrastive, args.wd_contrastive, args.optim_contrastive,
        args.feature_input_dim, args.feature_hidden_dim, args.feature_output_dim, args.feature_extractor_depth,
        args.projection_output_dim, 
        args.lr_unimodal, args.lr_bimodal, args.lr_trimodal,
        args.mm_unimodal, args.mm_bimodal, args.mm_trimodal,
        args.wd_unimodal, args.wd_bimodal, args.wd_trimodal,
        args.optim_unimodal, args.optim_bimodal, args.optim_trimodal,
        args.prediction_head_depth,
        args.num_epochs_contrastive, args.num_epochs_residual,  # Epoch and patience
        args.patience_contrastive, args.patience_residual, args.warmup_epochs,
        args.load_pretrain_contrastive, args.load_pretrain_residual, # Checkpoint
        args.load_checkpoint_filepath, 
        args.loss_fn_name, args.metric_list, args.seed, # Misc
    ]
    args_info = [str(info) for info in args_info]
    return args_info


def get_args_info_multitask(args) -> List[str]:
    args_info = [
        args.contrastive, # Training Setting
        args.in_batch_cl, args.batch_size_contrastive, # Hyperparameters
        args.lr_contrastive, args.mm_contrastive, args.wd_contrastive, args.optim_contrastive,
        args.feature_input_dim, args.feature_hidden_dim, args.feature_output_dim, args.feature_extractor_depth,
        args.projection_output_dim, 
        args.num_epochs_contrastive, args.patience_contrastive, # Epoch and patience
        args.seed, # Misc
        args.loop_multitask_times, # Multitask
        args.dataset_name_list, args.per_dataset_size, args.balanced,
    ]
    args_info = [str(info) for info in args_info]
    return args_info


def check_config_exist(args) -> bool:
    args_info = get_args_info(args)
    
    print(args_info)
    
    if os.path.exists(args.result_filepath):
        # check if the setting already exists
        df = pd.read_csv(args.result_filepath)
        for i in range(len(df)):
            entry = list(df.iloc[i][INDEX_COLUMNS])
            entry = [str(info) for info in entry]
            if entry == args_info:
                print("This setting already exists!")
                return True
    else:
        # create a new csv file
        csv_file = pd.DataFrame(columns=ALL_COLUMNS)
        csv_file.to_csv(args.result_filepath, index=False)
        print("Created a new csv file:", args.result_filepath)
    return False


def check_config_exist_multitask(args) -> bool:
    args_info = get_args_info_multitask(args)
    
    print(args_info)
    
    if os.path.exists(args.result_filepath):
        # check if the setting already exists
        df = pd.read_csv(args.result_filepath)
        for i in range(len(df)):
            entry = list(df.iloc[i][INDEX_COLUMNS_MULTITASK])
            entry = [str(info) for info in entry]
            if entry == args_info:
                print("This setting already exists!")
                return True
    else:
        # create a new csv file
        csv_file = pd.DataFrame(columns=ALL_COLUMNS_MULTITASK)
        csv_file.to_csv(args.result_filepath, index=False)
        print("Created a new csv file:", args.result_filepath)
    return False


def add_entry(args, metric_result: Dict[str, float]):
    args_info = get_args_info(args)
    
    # add entry
    df = pd.read_csv(args.result_filepath)
    df.loc[len(df)] = args_info + [args.save_checkpoint_filepath, metric_result]
    df.to_csv(args.result_filepath, index=False)
    
    print("Added entry to", args.result_filepath)
    

def add_entry_multitask(args, loss_contrastive: Dict[str, float]):
    args_info = get_args_info_multitask(args)
    
    # add entry
    df = pd.read_csv(args.result_filepath)
    df.loc[len(df)] = args_info + [args.save_checkpoint_filepath, loss_contrastive]
    df.to_csv(args.result_filepath, index=False)
    
    print("Added entry to", args.result_filepath)


def filter(args) -> bool:
    if args.lr_bimodal > args.lr_unimodal or args.lr_trimodal > args.lr_bimodal:
        print(f"Filtered args: {args.lr_unimodal}, {args.lr_bimodal}, {args.lr_trimodal}")
        return True
    return False