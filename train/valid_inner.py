from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import *
from train.utils import *
from train.loss import multi_modal_contrastive_loss


def valid_contrastive_inner(
    valid_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    projection_heads: Dict[str, ProjectionHead],
    args
) -> Dict[str, float]:
    all_modalities = valid_dataloader.dataset.all_modalities
    all_modality_pairs = get_pair_from_list(all_modalities)
    
    for mod, model in feature_extractors.items():
        model.eval()
    for mod, model in projection_heads.items():
        model.eval()
    
    valid_loss = {tuple(mod_pair): 0 for mod_pair in all_modality_pairs}
    with torch.no_grad():
        for data, label, _, task in tqdm(valid_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            projections = {mod: projection_heads[mod](feature_extractors[mod](data[mod])) 
                           for mod in all_modalities}
            
            for mod_pair in all_modality_pairs:
                loss = multi_modal_contrastive_loss(
                    projections[mod_pair[0]], projections[mod_pair[1]], label, task, args.in_batch_cl)
                valid_loss[mod_pair] += loss.item()
    
    valid_loss = {mod_pair: valid_loss[mod_pair] / len(valid_dataloader) for mod_pair in all_modality_pairs}
    if args.show_inner_logs:
        print(f'valid contrastive loss at inner iter: {valid_loss}')
    return valid_loss


def valid_unimodal_inner(
    valid_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Dict[str, Predictor],
    args,
) -> Tuple[float, Dict[str, float]]:
    all_modalities = valid_dataloader.dataset.all_modalities
    all_modality_pairs = get_pair_from_list(all_modalities)
    
    # Set model to eval mode
    for mod, model in feature_extractors.items():
        model.eval()
    for mod, model in unimodal_prediction_heads.items():
        model.eval()
    
    # Init
    loss_fn = get_loss_function(args.loss_fn_name)
    metric_function = get_metrix_function(args.metric_list)
    metric_result = zero_init_metric_result(args.metric_list)
    valid_loss = 0
    
    # Validation
    with torch.no_grad():
        for data, label, _, task in tqdm(valid_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            
            # forward
            features = {mod: feature_extractors[mod](data[mod]) for mod in data}
            unimodal_output = []
            for mod, model in unimodal_prediction_heads.items():
                unimodal_output.append(model(features[mod]))
            unimodal_pred = torch.mean(torch.stack(unimodal_output), dim=0)
            unimodal_loss = get_loss_function(args.loss_fn_name)(label, unimodal_pred)
            
            # update loss and metric
            valid_loss += unimodal_loss.item()
            for metric, func in metric_function.items():
                res = func(unimodal_pred, label)
                update_result(res, metric, metric_result, valid_dataloader.dataset.label_std)
            
    valid_loss /= len(valid_dataloader)
    metric_result = {metrics: metric_result[metrics] / len(valid_dataloader) for metrics in metric_result}
    if args.show_inner_logs:
        print(f'valid unimodal loss at inner iter: {valid_loss}')
        print(f'valid unimodal metric at inner iter: {metric_result}')
    return valid_loss, metric_result


def valid_bimodal_inner(
    valid_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Dict[str, Predictor],
    bimodal_prediction_heads: Dict[str, Predictor],
    args,
) -> Tuple[float, Dict[str, float]]:
    all_modalities = valid_dataloader.dataset.all_modalities
    all_modality_pairs = get_pair_from_list(all_modalities)
    
    # Set model to eval mode
    for mod, model in feature_extractors.items():
        model.eval()
    for mod, model in unimodal_prediction_heads.items():
        model.eval()
    for mod, model in bimodal_prediction_heads.items():
        model.eval()
    
    # Init
    loss_fn = get_loss_function(args.loss_fn_name)
    metric_function = get_metrix_function(args.metric_list)
    metric_result = zero_init_metric_result(args.metric_list)
    valid_loss = 0
    with torch.no_grad():
        for data, label, _, task in tqdm(valid_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            
            # forward
            features = {mod: feature_extractors[mod](data[mod]) for mod in data}
            
            if args.header_type in ['residual', 'uni_and_bi']:
                unimodal_output = []
                for mod, model in unimodal_prediction_heads.items():
                    unimodal_output.append(model(features[mod]))
                unimodal_pred = torch.mean(torch.stack(unimodal_output), dim=0)
            
            bimodal_output = []
            for mod_pair, model in bimodal_prediction_heads.items():
                bimodal_output.append(model(features[mod_pair[0]], features[mod_pair[1]]))
            bimodal_pred = torch.mean(torch.stack(bimodal_output), dim=0)
            if args.header_type in ['residual', 'uni_and_bi']:
                bimodal_pred = torch.sum(torch.stack([bimodal_pred, unimodal_pred]), dim=0)
            bimodal_loss = get_loss_function(args.loss_fn_name)(label, bimodal_pred)
            
            # update loss and metric
            valid_loss += bimodal_loss.item()
            for metric, func in metric_function.items():
                res = func(bimodal_pred, label)
                update_result(res, metric, metric_result, valid_dataloader.dataset.label_std)
            
    valid_loss /= len(valid_dataloader)
    metric_result = {metrics: metric_result[metrics] / len(valid_dataloader) for metrics in metric_result}
    if args.show_inner_logs:
        print(f'valid bimodal loss at inner iter: {valid_loss}')
        print(f'valid bimodal metric at inner iter: {metric_result}')
    return valid_loss, metric_result


def valid_trimodal_inner(
    valid_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Dict[str, Predictor],
    bimodal_prediction_heads: Dict[str, Predictor],
    trimodal_prediction_heads: Dict[str, Predictor],
    args,
) -> Tuple[float, Dict[str, float]]:
    all_modalities = valid_dataloader.dataset.all_modalities
    all_modality_pairs = get_pair_from_list(all_modalities)
    
    # Prepare
    for mod, model in feature_extractors.items():
        model.eval()
    for mod, model in unimodal_prediction_heads.items():
        model.eval()
    for mod, model in bimodal_prediction_heads.items():
        model.eval()
    for mod, model in trimodal_prediction_heads.items():
        model.eval()
    
    # Init
    loss_fn = get_loss_function(args.loss_fn_name)
    metric_function = get_metrix_function(args.metric_list)
    metric_result = zero_init_metric_result(args.metric_list)
    valid_loss = 0
    with torch.no_grad():
        for data, label, _, task in tqdm(valid_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            
            # forward
            features = {mod: feature_extractors[mod](data[mod]) for mod in data}
            
            if args.header_type == 'residual':
                unimodal_output = []
                for mod, model in unimodal_prediction_heads.items():
                    unimodal_output.append(model(features[mod]))
                unimodal_pred = torch.mean(torch.stack(unimodal_output), dim=0)
                
                bimodal_output = []
                for mod_pair, model in bimodal_prediction_heads.items():
                    bimodal_output.append(model(features[mod_pair[0]], features[mod_pair[1]]))
                bimodal_pred = torch.mean(torch.stack(bimodal_output), dim=0)
                bimodal_pred = torch.sum(torch.stack([bimodal_pred, unimodal_pred]), dim=0)
            
            trimodal_output = []
            for mod_tri, model in trimodal_prediction_heads.items():
                trimodal_output.append(model(features[mod_tri[0]], features[mod_tri[1]], features[mod_tri[2]]))
            trimodal_pred = torch.mean(torch.stack(trimodal_output), dim=0)
            if args.header_type == 'residual':
                trimodal_pred = torch.sum(torch.stack([trimodal_pred, bimodal_pred]), dim=0)
            trimodal_loss = get_loss_function(args.loss_fn_name)(label, trimodal_pred)
            
            # update loss and metric
            valid_loss += trimodal_loss.item()
            for metric, func in metric_function.items():
                res = func(trimodal_pred, label)
                update_result(res, metric, metric_result, valid_dataloader.dataset.label_std)
            
    valid_loss /= len(valid_dataloader)
    metric_result = {metrics: metric_result[metrics] / len(valid_dataloader) for metrics in metric_result}
    if args.show_inner_logs:
        print(f'valid trimodal loss at inner iter: {valid_loss}')
        print(f'valid trimodal metric at inner iter: {metric_result}')
    return valid_loss, metric_result
