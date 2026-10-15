import torch
from torch import nn


class ConvTransE(nn.Module):
    def __init__(self, dim, q_type='conv', static=False):
        super().__init__()
        self.q_type = q_type
        if q_type == 'conv':
            self.fc = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.BatchNorm1d(32),
                                    nn.ReLU(), nn.Dropout(0.2),
                                    nn.Conv1d(32, 1, 3 if static else 1,
                                              padding=1 if static else 0))
        else:
            linear = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
            if static:
                self.fc = linear
            else:
                self.w1 = linear
        if not static:
            self.pre_weight = nn.Parameter(torch.tensor(0.7))

    def query(self, subject_rows, condition_rows):
        """Call once for all rows in the complete direction; preserves BN and dropout."""
        if self.q_type == 'conv':
            return self.fc(torch.stack([subject_rows, condition_rows], dim=1)).squeeze(1)
        network = self.w1 if hasattr(self, 'w1') else self.fc
        return network(torch.cat([subject_rows, condition_rows], dim=1))

    def forward(self, embedding, relation, triples):
        return self.query(embedding[triples[:, 0]], relation[triples[:, 1]]) @ embedding.T
