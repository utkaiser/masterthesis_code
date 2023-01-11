import torch

from models.model_utils import sample_label_random

def a():
    b = torch.Tensor([1,2,3])
    print(b.repeat(2,1))


if __name__ == '__main__':
    a()