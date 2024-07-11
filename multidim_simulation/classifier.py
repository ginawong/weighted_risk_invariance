import torch.nn as nn
from torch import Tensor


class Classifier(nn.Module):
    def __init__(self, input_dims: int, num_classes: int):
        super().__init__()
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.feature_dims = max(num_classes, input_dims // 4)
        self.feature_model = nn.Sequential(
            nn.Linear(input_dims, max(num_classes, input_dims // 2)),
            nn.ReLU(),
            nn.Linear(max(num_classes, input_dims // 2), self.feature_dims),
            nn.ReLU(),)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dims, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc(self.features(x))
        return y

    def features(self, x: Tensor) -> Tensor:
        z = self.feature_model(x)
        return z


class TrueInvariantClassifier(Classifier):
    def __init__(self, input_dims: int, num_classes: int, r_dims: int):
        super().__init__(r_dims, num_classes)
        self.input_dims = input_dims
        self.r_dims = r_dims

    def forward(self, x):
        return super().forward(x[:, :self.r_dims])

    def features(self, x):
        return super().features(x[:, :self.r_dims])
