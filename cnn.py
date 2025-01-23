import torch.nn as nn
import torch.nn.functional as F

class ImprovedKeypointCNN(nn.Module):
    def __init__(self):
        super(ImprovedKeypointCNN, self).__init__()

        def make_stage(nin, nout, num_blocks, expand_ratio, stride):
            layers = []
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(MBConvBlock(nin, nout, expand_ratio, s))
                nin = nout
            return nn.Sequential(*layers)

        class MBConvBlock(nn.Module):
            def __init__(self, nin, nout, expand_ratio, stride):
                super(MBConvBlock, self).__init__()
                self.stride = stride
                self.use_residual = (stride == 1 and nin == nout)
                hidden_dim = nin * expand_ratio

                self.expand = nn.Conv2d(nin, hidden_dim, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(hidden_dim)

                self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
                self.bn2 = nn.BatchNorm2d(hidden_dim)

                self.project_conv = nn.Conv2d(hidden_dim, nout, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(nout)
            
            def forward(self, x):
                residual = x
                x = F.silu(self.bn1(self.expand(x)))
                x = F.silu(self.bn2(self.dw_conv(x)))
                x = self.bn3(self.project_conv(x))

                if self.use_residual:
                    x += residual
                return x
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            make_stage(32, 16, num_blocks=1, expand_ratio=1, stride=1),
            make_stage(16, 24, num_blocks=2, expand_ratio=6, stride=2),
            make_stage(24, 40, num_blocks=2, expand_ratio=6, stride=2),
            make_stage(40, 80, num_blocks=3, expand_ratio=6, stride=2)
        )

        self.head = nn.Sequential(
            nn.Conv2d(80, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 30)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        
        return x