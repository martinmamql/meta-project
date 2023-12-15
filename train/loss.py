from typing import Literal, List

import torch
from train.metrics import *


def correlation_loss(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    return torch.tensor(1) - correlation(y_true, y_hat)


def ccc_loss(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    return torch.tensor(1) - concordance_correlation_coefficient(y_true, y_hat)


def f1_loss(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    return torch.tensor(1) - weighted_one_vs_rest_f1(y_true, y_hat)[0]


def cross_entropy_loss(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    criterion = nn.BCEWithLogitsLoss()
    return criterion(y_hat, y_true)


def accuracy_loss(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    return torch.tensor(1) - accuracy(y_true, y_hat)


def multi_modal_contrastive_loss(
    projection_1: torch.tensor,
    projection_2: torch.tensor,
    labels: torch.tensor,
    task: Literal['R', 'C'],
    classes: List[int] = None,
    temperature: float = 0.5,
    width: float = 0.5,
    dtype = torch.float32,
    in_batch_cl: bool = True,
):
    sample_size = projection_1.size(0)
    
    norm_1 = torch.reshape(torch.norm(projection_1, dim=1), (-1, 1))
    norm_2 = torch.reshape(torch.norm(projection_2, dim=1), (1, -1))
    
    similarity_matrix = torch.div(
        torch.matmul(projection_1, torch.transpose(projection_2, 0, 1)),
        torch.matmul(norm_1, norm_2)
    ) / temperature
    exp_similarity = torch.exp(similarity_matrix).to(dtype)
    
    if not in_batch_cl:
        loss = 0
        nominator = exp_similarity.diagonal().to(dtype)
        denominator = torch.sum(exp_similarity, dim=1).to(dtype)
        loss = -torch.sum(torch.log(torch.div(nominator, demoninator)))
        loss /= sample_size
        return loss
    
    else:
        if task == 'R':
            loss = 0
            distance_matrix = torch.tensor(
                [[torch.abs(labels[i] - labels[j]) for j in range(sample_size)]
                for i in range(sample_size)]
            ).to(dtype).to(exp_similarity.device)
            
            distance_mean = torch.mean(distance_matrix, dim=1, keepdim=True)
            pos_selection = distance_matrix < distance_mean
            
            for i in range(sample_size):
                nominator = exp_similarity[i][pos_selection[i]].to(dtype)
                demoninator = torch.sum(exp_similarity[i]).to(dtype)
                # log_loss = -torch.sum(torch.log(torch.div(nominator, demoninator))) / \
                #             torch.sum(pos_selection[i])
                log_loss = -torch.log(torch.sum(torch.div(nominator, demoninator))) / \
                            torch.sum(pos_selection[i])
                loss += log_loss
            
            loss /= sample_size
            
            return loss
        
        else:
            loss = 0
            pos_selection = torch.tensor(
                [
                    [torch.all(torch.eq(labels[i], labels[j])) for j in range(sample_size)]
                    for i in range(sample_size)
                ], dtype=bool
            )
            
            for i in range(sample_size):
                nominator = exp_similarity[i][pos_selection[i]].to(dtype)
                demoninator = torch.sum(exp_similarity[i]).to(dtype)
                log_loss = -torch.sum(torch.log(torch.div(nominator, demoninator))) / \
                            torch.sum(pos_selection[i])
                loss += log_loss
            
            loss /= sample_size
            return loss