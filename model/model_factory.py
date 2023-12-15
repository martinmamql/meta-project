from typing import Dict, Literal, List, Optional

import torch
from torch.nn.parallel import DataParallel

from model.model import *
from train.utils import *
from train.metrics import *
from train.loss import *


def get_all_feature_extractors(args) -> Dict[str, Extractor]:
    return {
        mod: Extractor(
            input_dim=args.feature_input_dim[mod],
            hidden_dim=args.feature_hidden_dim[mod],
            output_dim=args.feature_output_dim[mod],
            layer_num=args.feature_extractor_depth,
        ) for mod in args.all_modalities
    }


def get_projection_heads(args) -> Dict[str, ProjectionHead]:
    return {
        mod: ProjectionHead(
            modality=mod,
            representation_dim=args.feature_output_dim[mod],
            projection_output_dim=args.projection_output_dim[mod],
        ) for mod in args.all_modalities
    }


def get_unimodal_models(args) -> Dict[str, UnimodalModel]:
    return {
        mod: UnimodalModel(
            modality=mod,
            prediction_head=Predictor(
                input_dim=args.feature_output_dim[mod],
                hidden_dim=int(1.5*args.feature_output_dim[mod]),
                output_dim=args.predictor_output_dim,
                layer_num=args.prediction_head_depth,
                softmax=False,
            ),
            classi_regressor=Classifier(
                input_size=args.predictor_output_dim,
                hidden_size=int(0.5*args.predictor_output_dim),
                num_classes=args.class_num,
            ) if args.task_type == 'C' else \
            Regressor(
                input_size=args.predictor_output_dim,
                hidden_size=int(0.5*args.predictor_output_dim),
            )
        ) for mod in args.all_modalities
    }
    

def get_bimodal_models(args) -> Dict[str, BimodalModel]:
    bimodal_dim = {mod_pair: sum([args.feature_output_dim[mod_pair[i]] for i in range(2)]) 
                   for mod_pair in get_pair_from_list(args.all_modalities)}
    return {
        mod_pair: BimodalModel(
            modalities=mod_pair,
            prediction_head=Predictor(
                input_dim=bimodal_dim[mod_pair],
                hidden_dim=int(1.5*bimodal_dim[mod_pair],),
                output_dim=bimodal_dim[mod_pair],
                layer_num=args.prediction_head_depth,
                softmax=False,
            ),
            classi_regressor=Classifier(
                input_size=bimodal_dim[mod_pair],
                hidden_size=int(0.5*bimodal_dim[mod_pair],),
                num_classes=args.class_num,
            ) if args.task_type == 'C' else \
            Regressor(
                input_size=bimodal_dim[mod_pair],
                hidden_size=int(0.5*bimodal_dim[mod_pair],),
            )
        ) for mod_pair in get_pair_from_list(args.all_modalities)
    }


def get_trimodal_models(args) -> Dict[str, TrimodalModel]:
    trimodal_dim = {mod_trip: sum([args.feature_output_dim[mod_trip[i]] for i in range(3)]) 
                    for mod_trip in get_triplet_from_list(args.all_modalities)}
    return {
        mod_trip: TrimodalModel(
            modalities=mod_trip,
            prediction_head=Predictor(
                input_dim=trimodal_dim[mod_trip],
                hidden_dim=int(1.5*trimodal_dim[mod_trip]),
                output_dim=trimodal_dim[mod_trip],
                layer_num=args.prediction_head_depth,
                softmax=False,
            ),
            classi_regressor=Classifier(
                input_size=trimodal_dim[mod_trip],
                hidden_size=int(0.5*trimodal_dim[mod_trip]),
                num_classes=args.class_num,
            ) if args.task_type == 'C' else \
            Regressor(
                input_size=trimodal_dim[mod_trip],
                hidden_size=int(0.5*trimodal_dim[mod_trip]),
            )
        ) for mod_trip in get_triplet_from_list(args.all_modalities)
    }
    

def get_all_optimizers(
    model: Dict[str, nn.Module],
    model_type: Literal['feature_extractor', 'projection_head', 'unimodal', 'bimodal', 'trimodal'],
    args
) -> Dict[str, torch.optim.Optimizer]:
    assert model_type in ['feature_extractor', 'projection_head', 'unimodal', 'bimodal', 'trimodal']
    mapping = {'feature_extractor': 'contrastive', 'projection_head': 'contrastive', 
               'unimodal': 'unimodal', 'bimodal': 'bimodal', 'trimodal': 'trimodal'}
    assert eval(f'args.optim_{mapping[model_type]}') in ['sgd', 'adam']
    
    lr = eval(f'args.lr_{mapping[model_type]}')
    mm = eval(f'args.mm_{mapping[model_type]}')
    wd = eval(f'args.wd_{mapping[model_type]}')
    optim = eval(f'args.optim_{mapping[model_type]}')
    
    return {mod: torch.optim.SGD(model[mod].parameters(), lr=lr, momentum=mm, weight_decay=wd) 
            if optim == 'sgd' else torch.optim.Adam(model[mod].parameters(), lr=lr, weight_decay=wd) 
            for mod in model.keys()}
    

def get_contrastive_modules(args) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]:
    # Models
    feature_extractors = get_all_feature_extractors(args)
    projection_heads = get_projection_heads(args)
    
    # Load pretrain checkpoint
    if args.load_pretrain_contrastive:
        assert args.load_checkpoint_filepath is not None
        checkpoint = torch.load(args.load_checkpoint_filepath)
        try:
            for mod, model in feature_extractors.items():
                model.load_state_dict(checkpoint['feature_extractors'][mod])
            for mod, model in projection_heads.items():
                model.load_state_dict(checkpoint['projection_heads'][mod])
            print(">>> Pretrain checkpoint for contrastive module loaded successfully.")
        except:
            print("Pretrain checkpoint model size does not match with current model size or checkpoint doesn't exist. Skip loading pretrain checkpoint.")
    
    for mod, model in feature_extractors.items():
        model = model.float()
        model = DataParallel(model)
        model.to(args.device)
    for mod, model in projection_heads.items():
        model = model.float()
        model = DataParallel(model)
        model.to(args.device)
    
    # Optimizers
    optimizer_feature = get_all_optimizers(feature_extractors, 'feature_extractor', args)
    optimizer_projection = get_all_optimizers(projection_heads, 'projection_head', args)
    
    return feature_extractors, projection_heads, optimizer_feature, optimizer_projection


def get_residual_models(args) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]:
    # Models
    unimodal_prediction_heads = get_unimodal_models(args) if args.header_type in ['unimodal', 'uni_and_bi', 'residual'] else {}
    bimodal_prediction_heads = get_bimodal_models(args) if args.header_type in ['bimodal', 'uni_and_bi', 'residual'] else {}
    trimodal_prediction_heads = get_trimodal_models(args) if args.header_type in ['trimodal', 'residual'] else {}
    
    # Load pretrain checkpoint
    if args.load_pretrain_residual:
        assert args.load_checkpoint_filepath is not None
        checkpoint = torch.load(args.load_checkpoint_filepath)
        try:
            for mod, model in unimodal_prediction_heads.items():
                model.load_state_dict(checkpoint['unimodal_prediction_heads'][mod])
            for mod, model in bimodal_prediction_heads.items():
                model.load_state_dict(checkpoint['bimodal_prediction_heads'][mod])
            for mod, model in trimodal_prediction_heads.items():
                model.load_state_dict(checkpoint['trimodal_prediction_heads'][mod])
            print(">>> Pretrain checkpoint for residual module loaded successfully.")
        except:
            print("Pretrain checkpoint model size does not match with current model size or checkpoint doesn't exist. Skip loading pretrain checkpoint.")
    
    for mod, model in unimodal_prediction_heads.items():
        model = model.float()
        model = DataParallel(model)
        model.to(args.device)
    for mod, model in bimodal_prediction_heads.items():
        model = model.float()
        model = DataParallel(model)
        model.to(args.device)
    for mod, model in trimodal_prediction_heads.items():
        model = model.float()
        model = DataParallel(model)
        model.to(args.device)
        
    # Optimizers
    unimodal_optimizers = get_all_optimizers(unimodal_prediction_heads, 'unimodal', args)
    bimodal_optimizers = get_all_optimizers(bimodal_prediction_heads, 'bimodal', args)
    trimodal_optimizers = get_all_optimizers(trimodal_prediction_heads, 'trimodal', args)
    
    return unimodal_prediction_heads, bimodal_prediction_heads, trimodal_prediction_heads, \
           unimodal_optimizers, bimodal_optimizers, trimodal_optimizers
