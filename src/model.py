import torch
from efficientnet_pytorch import EfficientNet

def get_efficient_net(load_path):
    model = EfficientNet.from_name('efficientnet-b3')
    model._fc = nn.Sequential(
        nn.Linear(in_features=1536, out_features=1, bias=True)
    )

    model.load_state_dict(torch.load(load_path))

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        model.cuda()

    return model