import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(785, 500),
            # nn.BatchNorm1d(200),
            # nn.Linear(200,500),
            # nn.ReLU(),
            nn.Linear(500, 43),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.clf(x)
        return x
