import torch
import torch.nn as nn

class Q_Network(nn.Module):
    """
    CNN for DQN
    """
    def __init__(self, img_stack, action_dim):
        super(Q_Network, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2), #349 266
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),       #173 132
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 4), stride=2), #86 65
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 1), stride=4),  #22 17
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(2, 1), stride=4), #6 5
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(4, 3), stride=2), #2 2
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=2),  #1 1
            nn.ReLU(),
        )
        self.output = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 256)
        x = self.output(x)
        return x
