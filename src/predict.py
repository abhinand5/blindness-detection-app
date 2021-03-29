import numpy as np

import torch 
from torch.autograd import Variable

import os 
from pathlib import Path

from model import get_efficient_net
from utils import round_off_preds, preprocess_image

MODEL_PATH = Path("./saved_models/best_val_kappa_score_026.pt")

class_mapper = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

def predict(img_path):
    model = get_efficient_net(MODEL_PATH)
    image = preprocess_image(img_path).to('cuda')

    input_image = Variable(image)
    input_image = input_image.to('cuda' if torch.cuda.is_available() else 'cpu')

    output = model(input_image[None, ...].float())
    output = round_off_preds(output.detach().cpu().numpy())[0][0]

    return class_mapper[output]


# if __name__ == "__main__":
#     for image in os.listdir('test_images'):
#         print(image)
#         print(predict(f'./test_images/{image}'))
#         print()