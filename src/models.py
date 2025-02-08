import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, out_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.LazyConv1d(out_channels, kernel_size=1)
        self.bn2 = nn.LazyBatchNorm1d()
        self.bn1 = nn.LazyBatchNorm1d()

        if downsample:
            self.conv2 = nn.LazyConv1d(out_channels, kernel_size=3, stride=2, padding=1)
            self.skip = nn.Sequential(*[nn.AvgPool1d(2), nn.LazyConv1d(out_channels, kernel_size=1)])
        else:
            self.conv2 = nn.LazyConv1d(out_channels, kernel_size=3, stride=1, padding=1)
            self.skip = nn.Identity() 

            
    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        Y += self.skip(x)
        return F.relu(Y)

class ResNetResidualBlock(nn.Module):
    def __init__(self, num_residuals, num_channels, downsample=False):
        super().__init__()
        blk = []
        for i in range(num_residuals):
            if i == 0 and downsample:
                blk.append(ResidualConvBlock(num_channels, downsample=True))
            else:
                blk.append(ResidualConvBlock(num_channels))

        self.block = nn.Sequential(*blk)

    def forward(self, x):
        return self.block(x)

class ResNetStemBlock(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv1d(out_channels, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x):
        return self.block(x)

class XRFAutoencoder(nn.Module):

    def __init__(self, latent_dim=10):
        super().__init__()
         
        self.encoder = nn.Sequential(
            ResNetStemBlock(64),
            *[ResNetResidualBlock(1, 64) for _ in range(2)],
            *[ResNetResidualBlock(2, 128 * 2**i, downsample=True) for i in range(2)],
            nn.LazyConv1d(latent_dim, 1)
        )

        self.classifiers = [nn.Sequential(*[nn.LazyLinear(128), nn.LazyLinear(128), nn.LazyLinear(1), nn.Sigmoid()]) for _ in range(latent_dim)]
        
        self.decoder = nn.Sequential(
        )
        
    def forward(self, x):
        z = self.encoder(x)
        z_prime = torch.stack([self.classifiers[0](z[:, c, :]) for c in range(z.shape[1])]).reshape(-1, z.shape[1])
        return z_prime
    
    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        for classifier in self.classifiers:
            classifier.to(device)
        return self 