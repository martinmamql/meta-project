# (weighted) F-1, Accuracy, Acc-7, Acc-2, 

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from scipy.special import expit, softmax
from scipy.stats import pearsonr, wilcoxon


def concordance_correlation_coefficient(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    """Calculate the concordance correlation coefficient(s)."""
    mean_y = torch.mean(y_true, dim=0)
    mean_y_hat = torch.mean(y_hat, dim=0)
    y_mean = y_true - mean_y
    y_hat_mean = y_hat - mean_y_hat
    cov = torch.mean(y_mean * y_hat_mean, dim=0)
    var = torch.var(y_true, dim=0, unbiased=False) + torch.var(y_hat, dim=0, unbiased=False)
    mse = (mean_y - mean_y_hat) ** 2
    ccc = (2 * cov) / (var + mse)
    return torch.mean(ccc)


def correlation(
    y_true: torch.tensor,
    y_hat: torch.tensor,
) -> torch.tensor:
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)
    correlation_matrix = torch.corrcoef(torch.stack([y_true, y_hat]))
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient


def mean_absolute_error(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.tensor:
    """Calculate the mean absolute error."""
    error = torch.abs(y_true - y_hat)
    return torch.mean(error, dim=0)


def mean_squared_error(
    y_true: torch.tensor,
    y_hat: torch.tensor,
) -> torch.tensor:
    criterion = nn.MSELoss()
    mse = criterion(y_true, y_hat)
    return mse


def weighted_one_vs_rest_f1(
    y_true_one_hot: torch.tensor,
    y_logit: torch.tensor,
) -> List[torch.tensor]:
    num_classes = y_logit.size(1)
    y_true = torch.argmax(y_true_one_hot, dim=1)
    y_pred = torch.argmax(y_logit, dim=1)
    
    precision_sum = torch.tensor(0, dtype=torch.float16, requires_grad=False)
    recall_sum = torch.tensor(0, dtype=torch.float16, requires_grad=False)
    f1_score_sum = torch.tensor(0, dtype=torch.float16, requires_grad=True)

    for class_idx in range(num_classes):
        class_logits = y_logit[:, class_idx]
        
        pred_target_binary = (y_pred == class_idx).float()
        label_target_binary = (y_true == class_idx).float()

        # Calculate true positives, false positives, and false negatives
        true_positives = torch.sum(pred_target_binary * label_target_binary)
        false_positives = torch.sum(pred_target_binary * (1 - label_target_binary))
        false_negatives = torch.sum((1 - pred_target_binary) * label_target_binary)

        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        precision_sum = precision * torch.sum(label_target_binary) + precision_sum
        recall_sum = recall * torch.sum(label_target_binary) + recall_sum
        f1_score_sum = f1 * torch.sum(label_target_binary) + f1_score_sum

    return [f1_score_sum / len(y_true), precision_sum / len(y_true), recall_sum / len(y_true)]

def accuracy(
    y_true: torch.Tensor,
    y_logit: torch.Tensor,
) -> torch.tensor:
    y_pred = torch.argmax(y_logit, dim=1)
    correct_predictions = torch.tensor(torch.eq(y_pred, torch.argmax(y_true, dim=1)), dtype=torch.float16, requires_grad=True)
    return correct_predictions.mean()
