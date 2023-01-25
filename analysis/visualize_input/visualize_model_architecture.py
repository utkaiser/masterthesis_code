from torchviz import make_dot
from models.model_components import model_unet
import torch

from torchsummary import summary

def vis():
    x = torch.randn(1, 4, 64 , 64).requires_grad_(True)
    model = model_unet.UNet(depth=3, wf=1, acti_func='relu', scale_factor=2)
    y = model(x)
    make_dot(y, params=dict(list(model_unet.named_parameters()))).render("torchviz", format="png")

def vis2():
    model = model_unet.UNet(depth=3, wf=1, acti_func='relu', scale_factor=2)
    summary(model, input_size=(4, 64 , 64))

if __name__ == '__main__':
    vis2()