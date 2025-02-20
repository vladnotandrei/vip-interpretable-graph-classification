import torch
from torch.nn import nn


class ClassifierMutagenicity(nn.Module):
    def __init__(self, query_size, tau=None):
        super().__init__()
        self.query_size = query_size
        self.output_dim = 2  # Num classes
        self.layer1 = nn.Linear(self.query_size, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.classifier = nn.Linear(1000, self.output_dim)

        self.tau = tau
        self.current_max = 0

        # self.norm1 = torch.nn.LayerNorm(1000)
        # self.norm2 = torch.nn.LayerNorm(1000)

        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        #x = self.relu(self.norm3(self.layer3(x)))  # Can try adding norm like this

        if self.tau == None:
            return self.classifier(x)

        else:
            query_logits = self.classifier(x)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_logits = query_logits + query_mask.cuda()

            query = self.softmax(query_logits / self.tau)

            query = (self.softmax(query_logits / 1e-9) - query).detach() + query
            return query

    def update_tau(self, tau):
        self.tau = tau


class QuerierMutagenicity(nn.Module):
    def __init__(self, query_size, tau=None):
        super().__init__()
        self.query_size = query_size
        self.output_dim = query_size
        self.layer1 = nn.Linear(self.query_size, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.classifier = nn.Linear(1000, self.output_dim)

        self.tau = tau
        self.current_max = 0

        # self.norm1 = torch.nn.LayerNorm(1000)
        # self.norm2 = torch.nn.LayerNorm(1000)

        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        #x = self.relu(self.norm3(self.layer3(x)))  # Can try adding norm like this

        if self.tau == None:
            return self.classifier(x)

        else:
            query_logits = self.classifier(x)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_logits = query_logits + query_mask.cuda()

            query = self.softmax(query_logits / self.tau)

            query = (self.softmax(query_logits / 1e-9) - query).detach() + query
            return query

    def update_tau(self, tau):
        self.tau = tau