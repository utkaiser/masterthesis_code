# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
# Forked from https://github.com/jvanvugt/pytorch-unet

from torch import nn

class UNet(nn.Module):

    def __init__(self, in_channels=4, n_classes=3, depth=3, wf=0, acti_func='relu', scale_factor=2):
        '''
            simple UNet Neural Network implementation

            params:
            in_channels: number of channels in input
            n_classes: number of channels in output
            depth: number of levels
            wf: channel multiplication factor each level
            acti_func: activation function
        '''

        super(UNet, self).__init__()
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i != 0:
                self.down_path.append(
                    nn.Conv2d(prev_channels,in_channels*2**(wf+i),kernel_size=3,stride=2,padding=1,bias=False))
                prev_channels = in_channels*2**(wf+i)
            self.down_path.append(
                UNetConvBlock(prev_channels,in_channels*2**(wf+i), acti_func)
            )
            prev_channels = in_channels*2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels,in_channels* 2 ** (wf + i), acti_func)
            )
            prev_channels =in_channels* 2 ** (wf + i)

        self.last = nn.ModuleList()
        for i in range(int(scale_factor/2)):
            self.last.append(nn.Upsample(mode='bilinear', scale_factor=2))
            self.last.append(UNetConvBlock(prev_channels, prev_channels, acti_func))
        self.last.append(nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=False))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            if i%2 == 0:
                blocks.append(down(x))

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-2])

        for i,layer in enumerate(self.last):
            x = layer(x)

        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, acti_func = 'identity'):

        if acti_func not in ['identity', 'relu']: raise ValueError("choose correct activation function")
        super(UNetConvBlock, self).__init__()

        if acti_func == 'identity':
            block = [
                nn.Conv2d(in_size, out_size, kernel_size=3, bias=False, padding=1),
                nn.Conv2d(out_size, out_size, kernel_size=3, bias=False, padding=1)
            ]
        else:
            block = [
                nn.Conv2d(in_size, out_size, kernel_size=3, bias=True, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
                nn.Conv2d(out_size, out_size, kernel_size=3, bias=True, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            ]

        self.block = nn.Sequential(*block)

    def forward(self, x): return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, acti_func='identity'):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2))
        self.conv_block = UNetConvBlock(in_size, out_size, acti_func)

    def forward(self, x, bridge): return self.conv_block(self.up(x)) + bridge
