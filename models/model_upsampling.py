import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


def choose_upsampling(mode,scale_factor=2):

    if mode == "UNet3":
        return UNet(wf=1, depth=3, scale_factor=scale_factor).double()
    elif mode == "UNet6":
        return UNet(wf=1, depth=6, scale_factor=scale_factor).double()
    elif mode == "Tiramisu":
        return Tiramisu(in_channels=4, scale_factor = scale_factor).double()
    elif mode == "UTransform":
        return UTransform(in_channels=4, scale_factor=scale_factor).double()
    elif mode == "Numerical_upsampling":
        return Numerical_upsampling(in_channels=4, scale_factor=scale_factor).double()
    else:
        raise NotImplementedError("This downsampling network has not been implemented yet!")


class UNet(nn.Module):
    '''
    JNet class.
    # Adapted from https://discuss.pytorch.org/t/unet-implementation/426
    # Forked from https://github.com/jvanvugt/pytorch-unet
    Init params
    in_channels: number of channels in input
    n_classes: number of channels in output
    depth: number of levels
    wf: channel multiplication factor each level
    acti_func: activation function
    '''

    def __init__(self, in_channels=4, n_classes=3, depth=3, wf=0, acti_func='relu', scale_factor=2):
        super(UNet, self).__init__()
        self.depth = depth
        prev_channels = in_channels
        self.acti_func = acti_func
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i != 0:
                self.down_path.append(
                    nn.Conv2d(prev_channels, in_channels * 2 ** (wf + i), kernel_size=3, stride=2, padding=1,
                              bias=False))
                prev_channels = in_channels * 2 ** (wf + i)
            self.down_path.append(
                UNetConvBlock(prev_channels, in_channels * 2 ** (wf + i), self.acti_func)
            )
            prev_channels = in_channels * 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, in_channels * 2 ** (wf + i), self.acti_func)
            )
            prev_channels = in_channels * 2 ** (wf + i)

        self.last = nn.ModuleList()
        for i in range(int(scale_factor / 2)):
            self.last.append(nn.Upsample(mode='bilinear', scale_factor=2))
            self.last.append(UNetConvBlock(prev_channels, prev_channels, self.acti_func))
        self.last.append(nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=False))

    def forward(self, x, skip_all=None):

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0:
                blocks.append(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])

        for i, layer in enumerate(self.last):
            if len(self.last) - 1 == i and torch.is_tensor(skip_all):
                x = skip_all + x  # skip connection is just addition right now TODO: is that ok? batchnorm? convolution?
            x = layer(x)
        return x


class UNetConvBlock(nn.Module):
    '''
    Convolution blocks
    '''

    def __init__(self, in_size, out_size, acti_func='identity'):
        super(UNetConvBlock, self).__init__()
        block = []

        if acti_func == 'identity':
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, bias=False, padding=1))
            block.append(nn.Conv2d(out_size, out_size, kernel_size=3, bias=False, padding=1))
        elif acti_func == 'relu':
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, bias=True, padding=1))
            block.append(nn.BatchNorm2d(out_size))
            block.append(nn.ReLU())
            block.append(nn.Conv2d(out_size, out_size, kernel_size=3, bias=True, padding=1))
            block.append(nn.BatchNorm2d(out_size))
            block.append(nn.ReLU())
        else:
            print('Choose either identity or relu \n')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    '''
    Upstream branch of JNet
    '''

    def __init__(self, in_size, out_size, acti_func='identity'):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2))
        self.conv_block = UNetConvBlock(in_size, out_size, acti_func)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.conv_block(up)

        out = out + bridge
        return out



class Tiramisu(nn.Module):
    '''
    source: https://github.com/bfortuner/pytorch_tiramisu
    '''

    def __init__(self, in_channels=4, down_blocks=(5,5), up_blocks=(5,5), bottleneck_layers=1,
                 growth_rate=16, out_chans_first_conv=48, n_classes=3, scale_factor = 2):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        skip_connection_channel_counts = []

        # First Convolution
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        # Downsampling path
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        # Bottleneck
        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        # Upsampling path
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))

            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        #Final DenseBlock
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)

        self.rescale = nn.ModuleList([])
        self.rescale.append(nn.Upsample(scale_factor=scale_factor, mode="bilinear"))

    def forward(self, x, skip_all=None):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)

            if i == len(self.up_blocks) - 1:
                out = self.rescale[0](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        return out #self.output_layer(out)


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d())

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList(
            [DenseLayer(in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)
        ])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out

class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]






class UTransform(nn.Module):
    def __init__(self, in_channels=4, classes=3, scale_factor=2):
        super(UTransform, self).__init__()
        # https://github.com/HXLH50K/U-Net-Transformer/blob/main/models/utransformer/U_Transformer.py
        # https://arxiv.org/pdf/2103.06104.pdf

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.MHSA = MultiHeadSelfAttention(512)
        self.up1 = TransformerUp(512, 256)
        self.up2 = TransformerUp(256, 128)

        if scale_factor == 2:
            self.up3 = TransformerUp(128, 64)
            self.up_sample = nn.ConvTranspose2d(64,128, kernel_size=2, stride=2)
            last = 128
        else:
            self.upsample1 = nn.ConvTranspose2d(128,256, kernel_size=2, stride=2)
            self.up3 = TransformerUp(256, 128)
            self.up_sample = nn.ConvTranspose2d(128,256, kernel_size=2, stride=2)
            last = 256

        self.outc = OutConv(last, classes)

    def forward(self, x, skip_all=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.MHSA(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        if self.scale_factor:
            x = self.upsample1(x)
        x = self.up3(x, x1)
        x = self.up_sample(x)
        return self.outc(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ])

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(*[
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        ])

    def forward(self, x):
        return self.maxpool_conv(x)

class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True))

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x

class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()

        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        # Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        V = self.value(S1)
        # Ype = self.positional_encoding_2d(Yc, Yh, Yw)
        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)
        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class Numerical_upsampling(torch.nn.Module):
    def __init__(self, in_channels=4, scale_factor=2):
        super(Numerical_upsampling, self).__init__()
        self.in_channel = in_channels
        self.scale_factor = scale_factor

    def forward(self, x):
        b, c, w, h = x.shape
        upsample_shape = w * self.scale_factor

        outputs = torch.zeros([b, c, upsample_shape, upsample_shape])
        outputs[:, 0, :, :] = F.upsample(x[:,0, :, :].unsqueeze(dim=0), size=(upsample_shape, upsample_shape), mode='bilinear')
        outputs[:, 1, :, :] = F.upsample(x[:,1, :, :].unsqueeze(dim=0), size=(upsample_shape, upsample_shape), mode='bilinear')
        outputs[:, 2, :, :] = F.upsample(x[:,2, :, :].unsqueeze(dim=0), size=(upsample_shape, upsample_shape), mode='bilinear')
        return outputs






