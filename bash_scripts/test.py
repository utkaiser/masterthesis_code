import torch

if __name__ == "__init__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available())