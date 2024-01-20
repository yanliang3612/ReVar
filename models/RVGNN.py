import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import reset
import torch_geometric.transforms as T

class RVGNN(nn.Module):
    def __init__(self, encoder, classifier, unique_labels, tau=0.1, thres=0.9, device=0):
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier

        self.tau = tau
        self.thres = thres

        self.softmax = nn.Softmax(dim=1)
        self.num_unique_labels = len(unique_labels)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.reset_parameters()
        
    def forward(self, x):
        rep = self.encoder(x)
        return rep

    def snn(self, query, supports, labels):
        query = F.normalize(query)
        supports = F.normalize(supports)

        return self.softmax(query @ supports.T / self.tau) @ labels

    def loss(self, anchor, pos, anchor_supports, pos_supports, labels, gt_labels, train_mask):

        with torch.no_grad():
            gt_labels = gt_labels[train_mask].unsqueeze(-1)
            matrix = torch.zeros(train_mask.sum().item(), self.num_unique_labels).to(self.device)
            gt_matrix = matrix.scatter_(1, gt_labels, 1)

        probs1 = self.snn(anchor, anchor_supports, labels)
        with torch.no_grad():
            targets1 = self.snn(pos, pos_supports, labels)
            values, _ = targets1.max(dim=1)
            boolean = torch.logical_or(values>self.thres, train_mask)
            indices1 = torch.arange(len(targets1))[boolean]
            targets1[targets1 < 1e-4] *= 0
            targets1[train_mask] = gt_matrix
            targets1 = targets1[indices1]

        probs1 = probs1[indices1]
        loss = torch.mean(torch.sum(torch.log(probs1**(-targets1)), dim=1))

        return loss

    def cls(self, x):
        out = self.encoder(x)
        return self.classifier(out)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.classifier)


