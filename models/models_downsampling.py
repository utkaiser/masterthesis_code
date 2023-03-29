import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def choose_downsampling(mode, res_scaler, model_res):

    if mode == "Simple":
        return Simple_restriction(res_scaler, model_res)
    elif mode == "CNN":
        return CNN_restriction(res_scaler).to(device)
    elif mode == "Interpolation":
        return Numerical_downsampling(res_scaler)
    else:
        raise NotImplementedError("This downsampling network has not been implemented yet!")


class Numerical_downsampling(torch.nn.Module):
    def __init__(self, res_scaler):
        super(Numerical_downsampling, self).__init__()
        self.res_scaler = res_scaler

    def forward(self, x):
        u_x, u_y, u_t_c, vel = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]  # b x w x h
        new_res = x.shape[-1] // self.res_scaler
        restr_output = torch.zeros([u_x.shape[0], 4, new_res, new_res])
        restr_output[:, 0, :, :] = F.upsample(u_x[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        restr_output[:, 1, :, :] = F.upsample(u_y[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        restr_output[:, 2, :, :] = F.upsample(u_t_c[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        restr_output[:, 3, :, :] = F.upsample(vel[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear')
        return restr_output, None


class Simple_restriction(torch.nn.Module):
    def __init__(self, res_scaler, model_res):
        super(Simple_restriction, self).__init__()
        self.res_scaler = res_scaler

        self.layer_size = 4 * model_res * model_res
        self.new_size = model_res // res_scaler

        self.block1 = torch.nn.Sequential(*[
            torch.nn.Linear(self.layer_size, self.layer_size),
            torch.nn.BatchNorm1d(self.layer_size),
            torch.nn.ReLU()
        ])
        self.block2 = torch.nn.Sequential(*[
            torch.nn.Linear(self.layer_size // self.res_scaler, self.layer_size // self.res_scaler),
            torch.nn.BatchNorm1d(self.layer_size // self.res_scaler),
            torch.nn.ReLU()
        ])
        self.lin3 = torch.nn.Linear(self.layer_size // self.res_scaler, self.layer_size // self.res_scaler)

    def forward(self,x):
        x = x.view(-1)
        x = self.block1(x)
        x = self.block2(x)
        skip = x.clone()
        x = self.lin3(x)
        return x.view(self.in_channels,self.new_size,self.new_size), skip


class CNN_restriction(torch.nn.Module):
    def __init__(self, res_scaler):
        super(CNN_restriction, self).__init__()
        in_channels = 4

        self.restr_layer1 = Restr_block(in_channels, in_channels * 2, groups=in_channels, stride=2, kernel=2).double()
        self.restr_layer2 = Restr_block(in_channels * 2, in_channels * 4, groups=in_channels, stride=2, kernel=2).double()
        self.restr_layer3 = Restr_block(in_channels * 4, in_channels * 8, groups=in_channels, stride=2, kernel=2).double()
        self.restr_layer4 = Up_block(in_channels * 8, in_channels * 4, stride = 2, groups=in_channels, kernel=2).double()
        self.restr_layer5 = Up_block(in_channels * 4, in_channels * 2, stride=2, relu=False, batch_norm=False,
                                     groups=in_channels, kernel=2, padding=0).double()
        self.restr_layer6 = Up_block(in_channels * 2, in_channels, relu = False, batch_norm = False, groups=in_channels).double()

    def forward(self, x):
        x = self.restr_layer1(x)
        x = self.restr_layer2(x)
        x = self.restr_layer3(x)
        x = self.restr_layer4(x)
        x = self.restr_layer5(x)
        x = self.restr_layer6(x)
        return x, None


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

class Up_block(torch.nn.Module):
    def __init__(self, in_channels, out_channel, stride=1, relu=True, batch_norm=True, kernel=3, padding=1, groups=1):
        super().__init__()

        layers = [
            torch.nn.ConvTranspose2d(in_channels, out_channel, kernel_size=kernel, padding=padding, stride=stride, groups=groups)
        ]
        if batch_norm: layers += [torch.nn.BatchNorm2d(out_channel)]
        if relu: layers += [torch.nn.ReLU()]

        self.restr = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.restr(x)


