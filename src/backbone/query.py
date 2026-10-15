import math
import torch
from torch import nn
import torch.nn.functional as F

class Qencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_sizes=[2,3], out_dim=256):
        super(Qencoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(hidden_dim * len(kernel_sizes), out_dim)

        self.w1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=int(math.floor(3 / 2))),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 1, kernel_size=3, stride=1,padding=int(math.floor(3 / 2)))
        )

    def forward(self, x, e1_emb):
        # x: [B, N, D] → permute to [B, D, N] for Conv1d
        x = x.permute(0, 2, 1)

        # apply multiple convolution filters and pool
        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))                 # [B, hidden_dim, N]
            p = F.adaptive_max_pool1d(c, 1)     # [B, hidden_dim, 1]
            conv_outputs.append(p.squeeze(-1))  # [B, hidden_dim]

        # concat all filters' outputs
        out = torch.cat(conv_outputs, dim=1)    # [B, hidden_dim * len(kernel_sizes)]
        out = self.fc(out)                      # [B, out_dim]
        query_emb = self.w1(torch.stack([e1_emb, out], dim=1)).squeeze(1)
        return query_emb


