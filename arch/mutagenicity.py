import torch
import torch.nn as nn


class ClassifierMutagenicity(nn.Module):
    def __init__(self, queryset_size):
        super().__init__()
        self.queryset_size = queryset_size
        self.output_dim = 2  # Num classes
        self.layer1 = nn.Linear(self.queryset_size, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.classifier = nn.Linear(1000, self.output_dim)

        # self.tau = tau
        # self.current_max = 0

        # self.norm1 = torch.nn.LayerNorm(1000)
        # self.norm2 = torch.nn.LayerNorm(1000)

        # activations
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        #x = self.relu(self.norm3(self.layer3(x)))  # Can try adding norm like this

        return self.classifier(x)
    
        # if self.tau == None:
        #     return self.classifier(x)
        # else:
        #     query_logits = self.classifier(x)
        #     query_mask = torch.where(mask == 1, -1e9, 0.)  # Why this -1e9?
        #     query_logits = query_logits + query_mask.cuda()

        #     query = self.softmax(query_logits / self.tau)

        #     query = (self.softmax(query_logits / 1e-9) - query).detach() + query
        #     return query

    # def update_tau(self, tau):
    #     self.tau = tau


class QuerierMutagenicity(nn.Module):
    def __init__(self, queryset_size, tau=1.0):
        super().__init__()
        self.queryset_size = queryset_size
        # self.output_dim = queryset_size
        self.layer1 = nn.Linear(self.queryset_size, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.classifier = nn.Linear(1000, self.queryset_size)

        self.tau = tau
        # self.current_max = 0

        # self.norm1 = torch.nn.LayerNorm(1000)
        # self.norm2 = torch.nn.LayerNorm(1000)

        # activations
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        #x = self.relu(self.norm3(self.layer3(x)))  # Can try adding norm like this

        # if self.tau == None:
        #     return self.classifier(x)

        # else:
        query_logits = self.classifier(x)
        query_mask = torch.where(mask == 1, -1e9, 0.).to(query_logits.device)  # Why this -1e9?
        query_logits = query_logits + query_mask#.cuda()

        query = self.softmax(query_logits / self.tau)  # straight-through trick

        query = (self.softmax(query_logits / 1e-9) - query).detach() + query  # Why?
        return query

    def update_tau(self, tau):
        self.tau = tau