import argparse, time, json, os

import torch

from train.info import *
from train.utils import loss_and_metric_for_dataset

def list_of_strings(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser(description='TMLR')
    
    # Data
    parser.add_argument('--dataset_name', type=str, choices=ALL_DATASETS)
    parser.add_argument('--dataset_dir', type=str, default='/work/siyuanwu/meta/')
    parser.add_argument('--dataset_split', type=list, default=[0])
    
    # Training Setting
    parser.add_argument('--contrastive', default=True, action='store_false')
    parser.add_argument('--header_type', type=str, default='residual', choices=['residual', 'unimodal', 'uni_and_bi', 'bimodal', 'trimodal'])
    parser.add_argument('--header_training_paradigm', type=str, default='routing', choices=['routing', 'simultaneous'])
    parser.add_argument('--in_batch_cl', default=True, action='store_false')
    parser.add_argument('--freeze_contrastive_param', default=True, action='store_false')
    parser.add_argument('--use_multitask_pretrain', default=False, action='store_true')
    
    # Hyperparameters
    parser.add_argument('--batch_size_contrastive', type=int, default=32)
    parser.add_argument('--batch_size_residual', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--num_epochs_contrastive', type=int, default=100)
    parser.add_argument('--num_epochs_residual', type=int, default=100)
    parser.add_argument('--patience_contrastive', type=int, default=10)
    parser.add_argument('--patience_residual', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    
    parser.add_argument('--lr_contrastive', type=float, default=1e-3)
    parser.add_argument('--mm_contrastive', type=float, default=0.2)
    parser.add_argument('--wd_contrastive', type=float, default=1e-6)
    parser.add_argument('--optim_contrastive', type=str, default='adam', choices=['adam', 'sgd'])
    
    parser.add_argument('--feature_input_dim', type=dict, default=MODALITY_FEATURE_SIZE)
    parser.add_argument('--feature_hidden_dim', type=dict, default=MODALITY_HIDDEN_DIM)
    parser.add_argument('--feature_output_dim', type=dict, default=MODALITY_FEATURE_SIZE)
    parser.add_argument('--feature_extractor_depth', type=int, default=3)
    parser.add_argument('--projection_output_dim', type=int, default=64)
    
    parser.add_argument('--lr_unimodal', type=float, default=1e-3)
    parser.add_argument('--lr_bimodal', type=float, default=1e-3)
    parser.add_argument('--lr_trimodal', type=float, default=1e-3)
    parser.add_argument('--mm_unimodal', type=float, default=0.2)
    parser.add_argument('--mm_bimodal', type=float, default=0.2)
    parser.add_argument('--mm_trimodal', type=float, default=0.2)
    parser.add_argument('--wd_unimodal', type=float, default=1e-6)
    parser.add_argument('--wd_bimodal', type=float, default=1e-6)
    parser.add_argument('--wd_trimodal', type=float, default=1e-6)
    parser.add_argument('--optim_unimodal', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--optim_bimodal', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--optim_trimodal', type=str, default='adam', choices=['adam', 'sgd'])
    
    parser.add_argument('--predictor_output_dim', type=int, default=64)
    parser.add_argument('--prediction_head_depth', type=int, default=3)
    
    # Checkpoint
    parser.add_argument('--load_pretrain_contrastive', default=False, action='store_true')
    parser.add_argument('--load_pretrain_residual', default=False, action='store_true')
    parser.add_argument('--save_checkpoint', default=True, action='store_false')
    parser.add_argument('--load_checkpoint_filepath', type=str, default=None)
    parser.add_argument('--save_checkpoint_filename', type=str, default=None)
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='/work/siyuanwu/tmlr/output/')
    parser.add_argument('--output_redirect', default=True, action='store_false')
    parser.add_argument('--redirect_filename', type=str, default=None)
    parser.add_argument('--save_result', default=True, action='store_false')
    parser.add_argument('--result_filename', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=None)
    
    # Multitask
    parser.add_argument('--multitask', default=False, action='store_true')
    parser.add_argument('--loop_multitask_times', type=int, default=1)
    parser.add_argument('--dataset_name_list', type=list_of_strings, default=ALL_DATASETS)
    parser.add_argument('--per_dataset_size', type=int, default=256)
    parser.add_argument('--balanced', default=True, action='store_false')
    parser.add_argument('--max_num', type=int, default=4)
    
    # Transfer
    parser.add_argument('--train_prediction_head_only', default=False, action='store_true')
    
    # Misc
    parser.add_argument('--seed', type=int, default=1706)
    parser.add_argument('--log_freq_contrastive', type=int, default=5)
    parser.add_argument('--log_freq_residual', type=int, default=20)
    parser.add_argument('--num_train_per_validation_contrastive', type=int, default=5)
    parser.add_argument('--num_train_per_validation_residual', type=int, default=20)
    parser.add_argument('--show_inner_logs', default=False, action='store_true')
    parser.add_argument('--filter', default=True, action='store_false')
    parser.add_argument('--duplicate_enable', default=False, action='store_true')
    
    # Computed args
    args = parser.parse_args()
    
    assert args.num_epochs_contrastive % args.log_freq_contrastive == 0
    assert args.num_epochs_residual % args.log_freq_residual == 0
    assert args.log_freq_contrastive % args.num_train_per_validation_contrastive == 0
    assert args.log_freq_residual % args.num_train_per_validation_contrastive == 0
    
    if args.use_multitask_pretrain:
        assert args.load_pretrain_contrastive is not None and args.load_checkpoint_filepath is not None
    
    # modality, task, class number and loss/metrics
    if not args.multitask:
        args.all_modalities = DATASET_MODALITY[args.dataset_name]
        args.task_type = DATASET_TASK[args.dataset_name]
        args.class_num = DATASET_CLASS_NUMBER[args.dataset_name]
        args.loss_fn_name, args.metric_list = loss_and_metric_for_dataset(args.dataset_name)
    else:
        args.all_modalities = ALL_MODALITIES
    
    # save checkpoint
    if args.save_result:
        assert args.save_checkpoint
    if args.save_checkpoint and args.save_checkpoint_filename is None:
        if args.multitask:
            args.save_checkpoint_filename = f"{int(time.time())}.pt"
            args.save_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint_multitask_pretrain')
            args.save_checkpoint_filepath = os.path.join(args.output_dir, 'checkpoint_multitask_pretrain', 
                                                         args.save_checkpoint_filename)
        else:
            args.save_checkpoint_filename = f"{args.dataset_name}_{int(time.time())}.pt"
            args.save_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
            args.save_checkpoint_filepath = os.path.join(args.output_dir, 'checkpoint', args.save_checkpoint_filename)
    
    # output redirect
    if args.output_redirect and args.redirect_filename is None:
        if args.multitask:
            args.redirect_filename = f"{int(time.time())}.log"
        else:
            args.redirect_filename = f"{args.dataset_name}_{int(time.time())}.log"
        args.redirect_filepath = os.path.join(args.output_dir, args.redirect_filename)
    
    # save result
    if args.save_result:
        if args.contrastive and args.header_type == 'residual' \
            and args.in_batch_cl and args.freeze_contrastive_param and not args.use_multitask_pretrain:
            train_setting = 'full_model'
        elif not args.contrastive and args.header_type == 'residual' \
            and args.in_batch_cl and not args.freeze_contrastive_param and not args.use_multitask_pretrain:
            train_setting = 'no-cl'
        elif args.contrastive and args.header_type == 'trimodal' \
            and args.in_batch_cl and args.freeze_contrastive_param and not args.use_multitask_pretrain:
            train_setting = 'no-residual'
        elif args.contrastive and args.header_type == 'residual' \
            and not args.in_batch_cl and args.freeze_contrastive_param and not args.use_multitask_pretrain:
            train_setting = 'no-in-batch-cl'
        elif args.contrastive and args.header_type == 'residual' \
            and args.in_batch_cl and not args.freeze_contrastive_param and not args.use_multitask_pretrain:
            train_setting = 'no-freeze-param'
        elif args.contrastive and args.in_batch_cl and args.use_multitask_pretrain:
            train_setting = 'multi-task-pretraining'
            
        elif args.contrastive and args.in_batch_cl and args.freeze_contrastive_param and not args.use_multitask_pretrain:
            training_setting = 'unimodal'
        else:
            raise NotImplementedError(f"train setting {'contrastive' if args.contrastive else 'no-cl'}, {args.header_type}, {'in-batch-cl' if args.in_batch_cl else 'no-in-batch-cl'}, {'freeze-param' if args.freeze_contrastive_param else 'no-freeze-param'}, {'multi-task-pretrain' if args.use_multitask_pretrain else 'no-multi-task-pretrain'}")
        
        if args.result_filename is None:
            args.result_filename = f"{args.dataset_name}.csv" if not args.multitask else f"multitask-{args.optim_contrastive}.csv"
        if args.result_dir is None:
            if args.multitask or args.header_type == 'trimodal' or args.header_type == 'unimodal':
                args.result_dir = os.path.join(args.output_dir, train_setting)
                args.result_filepath = os.path.join(args.output_dir, train_setting, args.result_filename)
            else:
                args.result_dir = os.path.join(args.output_dir, train_setting, args.header_training_paradigm)
                args.result_filepath = os.path.join(args.output_dir, train_setting, args.header_training_paradigm, args.result_filename)
        else:
            args.result_filepath = os.path.join(args.result_dir, args.result_filename)
        
    # projection output dim
    args.projection_output_dim = {mod: args.projection_output_dim for mod in ALL_MODALITIES}
    
    # device
    args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    return args