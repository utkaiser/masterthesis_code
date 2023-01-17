import torch
import torch.nn.functional as F


class Interpolation_net(torch.nn.Module):
    def __init__(self, sizing_factor=2):
        super(Interpolation_net, self).__init__()
        self.sizing_factor = sizing_factor

    def forward(self, x):
        u_x, u_y, u_t_c, vel = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]  # b x w x h
        new_res = x.shape[-1] // self.sizing_factor
        restr_output = torch.zeros([u_x.shape[0], 4, new_res, new_res])
        restr_output[:, 0, :, :] = F.upsample(u_x[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        restr_output[:, 1, :, :] = F.upsample(u_y[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        restr_output[:, 2, :, :] = F.upsample(u_t_c[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        restr_output[:, 3, :, :] = F.upsample(vel[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        return restr_output, None


class Simple_restriction(torch.nn.Module):
    def __init__(self, in_channels):
        super(Simple_restriction, self).__init__()
        self.in_channels = in_channels

        size = in_channels * 128 * 128

        self.block1 = torch.nn.Sequential(*[
            torch.nn.Linear(size, size),
            torch.nn.BatchNorm1d(size),
            torch.nn.ReLU()
        ])
        self.block2 = torch.nn.Sequential(*[
            torch.nn.Linear(size // 2, size // 2),
            torch.nn.BatchNorm1d(size // 2),
            torch.nn.ReLU()
        ])
        self.lin3 = torch.nn.Linear(size // 2, size // 2)

    def forward(self,x):
        x = x.view(-1)
        x = self.block1(x)  # 128
        x = self.block2(x)  # 64
        skip = x.clone()
        x = self.lin3(x)  # 64
        return x.view(self.in_channels,64,64), skip


class CNN_restriction(torch.nn.Module):
    def __init__(self, in_channels):
        super(CNN_restriction, self).__init__()

        self.restr_layer1 = Restr_block(in_channels, in_channels * 2, groups=in_channels).double()
        self.restr_layer3 = Restr_block(in_channels * 2, in_channels * 2, groups=in_channels).double()
        self.restr_layer5 = Restr_block(in_channels * 2, in_channels * 2, groups=in_channels,stride=2).double()
        self.restr_layer8 = Restr_block(in_channels * 2, in_channels, relu=False, batch_norm=False,
                                        groups=in_channels).double()

    def forward(self, x):
        x = self.restr_layer1(x)  # 128
        x = self.restr_layer3(x)  # 64
        skip = x.clone()
        x = self.restr_layer5(x)  # 64
        x = self.restr_layer8(x)  # 64
        return x, skip


class Restr_block(torch.nn.Module):
    def __init__(self, in_channels, out_channel, stride=1, relu=True, batch_norm=True, kernel=3, padding=1, groups=1):
        super().__init__()

        layers = [
            torch.nn.Conv2d(in_channels, out_channel, kernel_size=kernel, padding=padding, stride = stride, groups=groups)
        ]
        if batch_norm: layers += [torch.nn.BatchNorm2d(out_channel)]
        if relu: layers += [torch.nn.ReLU()]

        self.restr = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.restr(x)


# outputs = torch.zeros([wx.shape[0], 3, 128, 128])
# outputs[:, 0, :, :] = F.upsample(wx[:, :, :].unsqueeze(dim=0), size=(128, 128), mode='bilinear')
# outputs[:, 1, :, :] = F.upsample(wy[:, :, :].unsqueeze(dim=0), size=(128, 128), mode='bilinear')
# outputs[:, 2, :, :] = F.upsample(wtc[:, :, :].unsqueeze(dim=0), size=(128, 128), mode='bilinear')