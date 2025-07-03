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

        # activations
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.classifier(x)


class QuerierMutagenicity(nn.Module):
    def __init__(self, queryset_size, tau=1.0):
        super().__init__()
        self.queryset_size = queryset_size
        self.layer1 = nn.Linear(self.queryset_size, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.classifier = nn.Linear(1000, self.queryset_size)

        self.tau = tau

        # activations
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        query_logits = self.classifier(x)
        query_mask = torch.where(mask == 1, -1e9, 0.).to(query_logits.device)  # -1,000,000,000 where mask==1, otherwise 0
        query_logits = query_logits + query_mask#.cuda()  # unmasked values will have very negative values

        # Straight-through softmax
        query = self.softmax(query_logits / self.tau)
        query = (self.softmax(query_logits / 1e-9) - query).detach() + query  # straight-through trick
        # ^ 1e-9 is temperature of softmax we're using here
        # Smaller temperature --> closer to hard argmax
        # from 1e-4 and smaller it seems to become hard argmax

        return query  # one hot vector of next-most informative query

    def update_tau(self, tau):
        self.tau = tau