import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
# ===== MODEL =====
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)