import torch
import torch.nn as nn

class BleedNet(nn.Module):
    def __init__(self, activation='relu'):
        super(BleedNet, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0),
            self.activation,
            nn.AdaptiveMaxPool2d(250),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.AdaptiveMaxPool2d(50),
        )
        
        self.wrap_up = nn.Sequential(
            nn.Linear(64 * 50 * 50, 512),
            self.activation,
            nn.Linear(512, 2),
        )

    def forward(self, x):
        out = self.layer1(x)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = out.reshape(out.size(0), -1)
        #print(out.size())
        out = self.wrap_up(out)
        #print(out.size())
        return out
    

class BleedNet2(nn.Module):
    def __init__(self, activation='relu'):
        super(BleedNet2, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=1, padding=0),
            self.activation,
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.AdaptiveMaxPool2d(50),
        )
        
        self.wrap_up = nn.Sequential(
            nn.Linear(64 * 50 * 50, 512),
            self.activation,
            nn.Linear(512, 2),
        )

    def forward(self, x):
        out = self.layer1(x)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = out.reshape(out.size(0), -1)
        #print(out.size())
        out = self.wrap_up(out)
        #print(out.size())
        return out

if __name__ == '__main__':
    import numpy as np
    net = BleedNet2()
    print('Batch!')
    fake_input = np.random.rand(3, 1, 512, 512) #5 batches, 1channel, 500*500 box
    fake_input = torch.from_numpy(fake_input).float()
    yhat = net.forward(fake_input)
    print('Done',yhat)
    
