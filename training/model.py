import torch
import torch.nn as nn
import torch.nn.functional as F

class BlackjackNet(nn.Module):
    def __init__(self):
        super(BlackjackNet, self).__init__()
        
        self.fc1 = nn.Linear(3, 32) # input: player total, dealer card, is soft hand
        self.fc2 = nn.Linear(32, 32) # hidden layer

        # head 1: win probability
        self.win_head = nn.Linear(32, 1)

        # head 2: action recommendation
        self.action_head = nn.Linear(32, 2)  # hit/stand

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        win_prob = torch.sigmoid(self.win_head(x))  # output in [0,1]
        action_logits = self.action_head(x)        # classification raw logits
        
        return win_prob, action_logits
