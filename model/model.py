from typing import List, Dict, Union, Tuple

import torch
from torch import nn

from train.info import DATASET_TASK, ALL_MODALITIES


class Regressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Regressor, self).__init__()
        self.input_dim = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.input_dim = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class Resnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, softmax=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = max(0, layer_num - 2)
        self.linear1 = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        
        self.has_softmax = softmax
        if softmax:
            self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.linear1(x)
        for _ in range(self.hidden_layers):
            x = x + self.linear2(x)
        x = self.linear3(x)
        return self.softmax(x) if self.has_softmax else x


class Predictor(Resnet):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num = 3, softmax=True):
        super().__init__(input_dim, hidden_dim, output_dim, layer_num, softmax)


class Extractor(Resnet):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=3, softmax=False):
        super().__init__(input_dim, hidden_dim, output_dim, layer_num, softmax)

    
class ProjectionHead(nn.Module):
    def __init__(
        self, 
        modality: str,
        representation_dim: int,
        projection_output_dim: int,
    ):
        super().__init__()
        self.modality = modality
        self.projection_head = nn.Sequential(
            nn.BatchNorm1d(num_features=representation_dim),
            nn.Linear(in_features=representation_dim, out_features=2*representation_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=2*representation_dim),
            nn.Linear(in_features=2*representation_dim, out_features=projection_output_dim),
            nn.BatchNorm1d(num_features=projection_output_dim),
        )
        
    def forward(self, x):
        return self.projection_head(x)


class UnimodalModel(nn.Module):
    def __init__(
        self, 
        modality: str,
        prediction_head: Predictor,
        classi_regressor: Union[Classifier, Regressor],
    ):
        super().__init__()
        self.modality = modality
        self.prediction_head = prediction_head
        self.classi_regressor = classi_regressor
        
    def forward(self, x):
        return self.classi_regressor(self.prediction_head(x))
    

class BimodalModel(nn.Module):
    def __init__(
        self, 
        modalities: Tuple[str],
        prediction_head: Predictor,
        classi_regressor: Union[Classifier, Regressor],
    ):
        super().__init__()
        assert len(modalities) == 2
        
        self.modalities = modalities
        self.prediction_head = prediction_head
        self.classi_regressor = classi_regressor

    def forward(self, feature_1, feature_2):
        feature = torch.cat([feature_1, feature_2], dim=-1)
        return self.classi_regressor(self.prediction_head(feature))
    
    
class TrimodalModel(nn.Module):
    def __init__(
        self, 
        modalities: Tuple[str],
        prediction_head: Predictor,
        classi_regressor: Union[Classifier, Regressor],
    ):
        super().__init__()
        assert len(modalities) == 3
        
        self.modalities = modalities
        self.prediction_head = prediction_head
        self.classi_regressor = classi_regressor

    def forward(self, feature_1, feature_2, feature_3):
        feature = torch.cat([feature_1, feature_2, feature_3], dim=-1)
        return self.classi_regressor(self.prediction_head(feature))

