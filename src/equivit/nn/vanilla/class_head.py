import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, dim: int, num_logits: int, drop_rate: float=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.drop_rate = drop_rate

        self.head = nn.Linear(dim, num_logits)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (*, L+1, C)        
        x = self.norm(x)

        # Pluck out class token
        x = x[..., -1, :]

        if self.drop_rate > 0.:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        
        x = self.head(x)
        return x
