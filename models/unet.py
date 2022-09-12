from torch import nn
import warnings
warnings.filterwarnings("ignore")

class UNet(nn.Module):
    '''
    UNet implementation
    adapted from https://discuss.pytorch.org/t/unet-implementation/426
    forked from https://github.com/jvanvugt/pytorch-unet

    params:
        in_channels: number of channels in input
        n_classes: number of channels in output
        depth: number of levels
        wf: channel multiplication factor each level (spatial extent of the filters)
    '''

    def __init__(self, in_channels=4, n_classes=3, depth=3, wf=0, scale_factor=4):
        super(UNet, self).__init__()

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i!=0:
                self.down_path.append(
                    nn.Conv2d(prev_channels,in_channels*2**(wf+i),kernel_size=3,stride=2,padding=1,bias=False))
                prev_channels = in_channels*2**(wf+i)
            self.down_path.append(
                UNetConvBlock(prev_channels,in_channels*2**(wf+i))
            )
            prev_channels = in_channels*2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels,in_channels* 2 ** (wf + i))
            )
            prev_channels =in_channels* 2 ** (wf + i)

        self.last = nn.ModuleList()
        for i in range(int(scale_factor/2)):
            self.last.append(nn.Upsample(mode='bilinear', scale_factor=2))
            self.last.append(UNetConvBlock(prev_channels, prev_channels))
        self.last.append(nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=False))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i%2 == 0:
                blocks.append(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-2])

        for i,layer in enumerate(self.last):
            x = layer(x)

        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetConvBlock, self).__init__()
        block = [
            nn.Conv2d(in_size, out_size, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        ]
        self.net = nn.Sequential(*block)

    def forward(self, x):
        return self.net(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2))
        self.conv_block = UNetConvBlock(in_size, out_size)

    def forward(self, x, bridge):
        return self.conv_block(self.up(x)) + bridge
