from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import *
from train.utils import *
from train.loss import multi_modal_contrastive_loss
from train.info import ALL_MODALITIES


def train_contrastive_inner(
    train_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    projection_heads: Dict[str, ProjectionHead],
    feature_optimizers: Dict[str, torch.optim.Optimizer],
    projection_optimizers: Dict[str, torch.optim.Optimizer],
    args,
) -> None:
    all_modalities = train_dataloader.dataset.all_modalities
    all_modality_pairs = get_pair_from_list(all_modalities)
    
    for mod, model in feature_extractors.items():
        model.train()
    for mod, model in projection_heads.items():
        model.train()
    for training_epoch in range(args.num_train_per_validation_contrastive):
        loss_epoch = {mod_pair: 0 for mod_pair in all_modality_pairs}
        
        # test
        # for data, label, _, task in tqdm(train_dataloader):
        #     print(data.keys())
        #     exit(0)
        # exit(0)
        # test
        
        for data, label, _, task in tqdm(train_dataloader):
            # all_modalities = list(data.keys())
            # all_modality_pairs = get_pair_from_list(all_modalities)
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            projections = {mod: projection_heads[mod](feature_extractors[mod](data[mod])) 
                           for mod in all_modalities}
            
            for mod_pair in all_modality_pairs:
                loss = multi_modal_contrastive_loss(
                    projections[mod_pair[0]], projections[mod_pair[1]], label, task, args.in_batch_cl)
                loss.backward(retain_graph=True)
                projection_optimizers[mod_pair[0]].step()
                projection_optimizers[mod_pair[1]].step()
                feature_optimizers[mod_pair[0]].step()
                feature_optimizers[mod_pair[1]].step()
                
                projection_optimizers[mod_pair[0]].zero_grad()
                projection_optimizers[mod_pair[1]].zero_grad()
                feature_optimizers[mod_pair[0]].zero_grad()
                feature_optimizers[mod_pair[1]].zero_grad()
                loss_epoch[mod_pair] += loss.item()
        loss_epoch = {mod_pair: loss_epoch[mod_pair] / len(train_dataloader) 
                        for mod_pair in all_modality_pairs}
        if args.show_inner_logs:
            print(f'train contrastive loss at inner iter {training_epoch}: {loss_epoch}')
        

def train_unimodal_inner(
    train_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Dict[str, Predictor],
    feature_extractor_optimizers: Dict[str, torch.optim.Optimizer],
    unimodal_optimizers: Dict[str, torch.optim.Optimizer],
    args,
) -> None:
    # Prepare
    if args.freeze_contrastive_param:
        for mod, model in feature_extractors.items():
            model.eval()
    else:
        for mod, model in feature_extractors.items():
            model.train()
    for mod, model in unimodal_prediction_heads.items():
        model.train()
        
    for training_epoch in range(args.num_train_per_validation_residual):
        loss_epoch = 0
        for data, label, _, task in tqdm(train_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            features = {mod: feature_extractors[mod](data[mod]) for mod in data}
            
            # forward
            unimodal_output = []
            for mod, model in unimodal_prediction_heads.items():
                unimodal_output.append(model(features[mod]))
            unimodal_pred = torch.mean(torch.stack(unimodal_output), dim=0)
            unimodal_loss = get_loss_function(args.loss_fn_name)(label, unimodal_pred)
            
            # backward
            unimodal_loss.backward()
            for mod, optim in unimodal_optimizers.items():
                optim.step()
                optim.zero_grad()
            if not args.freeze_contrastive_param:
                for mod, optim in feature_extractor_optimizers.items():
                    optim.step()
                    optim.zero_grad()
                
            loss_epoch += unimodal_loss.item()
        loss_epoch /= len(train_dataloader)
        if args.show_inner_logs:
            print(f'train unimodal loss at inner iter {training_epoch}: {loss_epoch}')
        

def train_bimodal_inner(
    train_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Dict[str, Predictor],
    bimodal_prediction_heads: Dict[str, Predictor],
    feature_extractor_optimizers: Dict[str, torch.optim.Optimizer],
    bimodal_optimizers: Dict[str, torch.optim.Optimizer],
    args,
) -> None:
    # Prepare
    if args.freeze_contrastive_param:
        for mod, model in feature_extractors.items():
            model.eval()
    else:
        for mod, model in feature_extractors.items():
            model.train()
    for mod, model in unimodal_prediction_heads.items():
        model.eval()
    for mod, model in bimodal_prediction_heads.items():
        model.train()
        
    for training_epoch in range(args.num_train_per_validation_residual):
        loss_epoch = 0
        for data, label, _, task in tqdm(train_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            features = {mod: feature_extractors[mod](data[mod]) for mod in data}
            
            # forward
            #? will it lead to memory leak without with torch.no_grad()?
            if args.header_type in ['residual', 'uni_and_bi']:
                with torch.no_grad():
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
            
            # backward
            bimodal_loss.backward()
            for mod, optim in bimodal_optimizers.items():
                optim.step()
                optim.zero_grad()
            if not args.freeze_contrastive_param:
                for mod, optim in feature_extractor_optimizers.items():
                    optim.step()
                    optim.zero_grad()
                
            loss_epoch += bimodal_loss.item()
        loss_epoch /= len(train_dataloader)
        if args.show_inner_logs:
            print(f'train bimodal loss at inner iter {training_epoch}: {loss_epoch}')


def train_trimodal_inner(
    train_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Dict[str, Predictor],
    bimodal_prediction_heads: Dict[str, Predictor],
    trimodal_prediction_heads: Dict[str, Predictor],
    feature_extractor_optimizers: Dict[str, torch.optim.Optimizer],
    trimodal_optimizers: Dict[str, torch.optim.Optimizer],
    args,
) -> None:
    # Prepare
    if args.freeze_contrastive_param:
        for mod, model in feature_extractors.items():
            model.eval()
    else:
        for mod, model in feature_extractors.items():
            model.train()
    for mod, model in unimodal_prediction_heads.items():
        model.eval()
    for mod, model in bimodal_prediction_heads.items():
        model.eval()
    for mod, model in trimodal_prediction_heads.items():
        model.train()
        
    for training_epoch in range(args.num_train_per_validation_residual):
        loss_epoch = 0
        for data, label, _, task in tqdm(train_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            
            # forward
            #? will it lead to memory leak without with torch.no_grad()?
            features = {mod: feature_extractors[mod](data[mod]) for mod in data}
            
            if args.header_type == 'residual':
                with torch.no_grad():
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
            
            # backward
            trimodal_loss.backward()
            for mod, optim in trimodal_optimizers.items():
                optim.step()
                optim.zero_grad()
            if not args.freeze_contrastive_param:
                for mod, optim in feature_extractor_optimizers.items():
                    optim.step()
                    optim.zero_grad()
                    
            loss_epoch += trimodal_loss.item()
        loss_epoch /= len(train_dataloader)
        if args.show_inner_logs:
            print(f'train trimodal loss at inner iter {training_epoch}: {loss_epoch}')
        

def train_simultaneous_inner(
    train_dataloader: DataLoader,
    feature_extractors: Dict[str, Extractor],
    unimodal_prediction_heads: Dict[str, Predictor],
    bimodal_prediction_heads: Dict[str, Predictor],
    trimodal_prediction_heads: Dict[str, Predictor],
    feature_extractor_optimizers: Dict[str, torch.optim.Optimizer],
    unimodal_optimizers: Dict[str, torch.optim.Optimizer],
    bimodal_optimizers: Dict[str, torch.optim.Optimizer],
    trimodal_optimizers: Dict[str, torch.optim.Optimizer],
    args,
) -> None:
    # Prepare
    if args.freeze_contrastive_param:
        for mod, model in feature_extractors.items():
            model.eval()
    else:
        for mod, model in feature_extractors.items():
            model.train()
    for mod, model in unimodal_prediction_heads.items():
        model.train()
    for mod, model in bimodal_prediction_heads.items():
        model.train()
    for mod, model in trimodal_prediction_heads.items():
        model.train()
        
    for training_epoch in range(args.num_train_per_validation_residual):
        loss_epoch = 0
        for data, label, _, task in tqdm(train_dataloader):
            data = {mod: feature.to(args.device) for mod, feature in data.items()}
            label = label.to(args.device)
            
            # forward
            #? will it lead to memory leak without with torch.no_grad() and keeping retrain_graph=True?
            features = {mod: feature_extractors[mod](data[mod]) for mod in data}

            # unimodal
            unimodal_output = []
            for mod, model in unimodal_prediction_heads.items():
                unimodal_output.append(model(features[mod]))
            unimodal_pred = torch.mean(torch.stack(unimodal_output), dim=0)
            detached_unimodal_pred = unimodal_pred.detach().clone().requires_grad_(False)
            unimodal_loss = get_loss_function(args.loss_fn_name)(label, unimodal_pred)
            unimodal_loss.backward(retain_graph=True)
            for mod, optim in unimodal_optimizers.items():
                optim.step()
                optim.zero_grad()
            
            # bimodal
            bimodal_output = []
            for mod_pair, model in bimodal_prediction_heads.items():
                bimodal_output.append(model(features[mod_pair[0]], features[mod_pair[1]]))
            bimodal_pred = torch.mean(torch.stack(bimodal_output), dim=0)
            bimodal_pred = torch.sum(torch.stack([bimodal_pred, detached_unimodal_pred]), dim=0)
            detached_bimodal_pred = bimodal_pred.detach().clone().requires_grad_(False)
            bimodal_loss = get_loss_function(args.loss_fn_name)(label, bimodal_pred)
            bimodal_loss.backward(retain_graph=True)
            for mod, optim in bimodal_optimizers.items():
                optim.step()
                optim.zero_grad()
                
            # trimodal
            trimodal_output = []
            for mod_tri, model in trimodal_prediction_heads.items():
                trimodal_output.append(model(features[mod_tri[0]], features[mod_tri[1]], features[mod_tri[2]]))
            trimodel_pred = torch.mean(torch.stack(trimodal_output), dim=0)
            trimodal_pred = torch.sum(torch.stack([trimodel_pred, detached_bimodal_pred]), dim=0)
            trimodal_loss = get_loss_function(args.loss_fn_name)(label, trimodal_pred)
            trimodal_loss.backward(retain_graph=True)
            for mod, optim in trimodal_optimizers.items():
                optim.step()
                optim.zero_grad()
                
            loss_epoch += trimodal_loss.item()
        loss_epoch /= len(train_dataloader)
        if args.show_inner_logs:
            print(f'train simultaneous residual loss at inner iter {training_epoch}: {loss_epoch}')
