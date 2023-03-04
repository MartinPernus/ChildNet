import argparse
import os

import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from PIL import Image

from models.childnet import ChildNet

cwd = os.path.dirname(__file__)

def read_image(path, unsqueeze=True):
    path = os.path.expanduser(path)
    img = to_tensor(Image.open(path).convert('RGB'))
    if unsqueeze:
        img = img.unsqueeze(0)
    return img

def int2tensor(x):
    return torch.tensor(x).view(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', choices=['nokdb', 'fiw'], default='nokdb')
    parser.add_argument('--father', type=str, default='imgs/father.jpg')
    parser.add_argument('--mother', type=str, default='imgs/mother.jpg')
    parser.add_argument('--child_real', type=str, default='imgs/child_real.jpg', help='only required for comparison purposes')
    parser.add_argument('--child_age', type=int, default=None, help='target child age class. Range [0, 9].')
    parser.add_argument('--child_gender', type=int, default=None, help='target child gender class. Range [0, 1].')
    parser.add_argument('--sample', action='store_true', help='enables child image sampling per fixed input')
    parser.add_argument('--move2parent', type=float, default=None, help="factor for dominant parent image. Range [-1, 1]")
    parser.add_argument('--cuda_idx', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parse_args()

    device = f'cuda:{args.cuda_idx}'

    model = ChildNet(args.model_weights, sample=args.sample).to(device)

    img_father = read_image(args.father).to(device)
    img_mother = read_image(args.mother).to(device)

    child_age = args.child_age
    child_gender = args.child_gender

    if child_age is not None and child_gender is not None:
        child_age = int2tensor(child_age).to(device)
        child_gender = int2tensor(child_gender).to(device)

    img_child = model(img_father, img_mother, child_age, child_gender, args.move2parent)
    save_image(img_child, 'imgs/result.jpg')
