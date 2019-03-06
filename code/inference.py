import random
import torch
import torchvision
import numpy as np
import warnings
from torch.utils.data import DataLoader
import config
from dataset import TestDataset
from utils.utils import *
import json
import os
import argparse

# 1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_submit_result(preds, image_names):
    assert len(preds) == len(image_names)
    data = dict()
    for pred, image_name in zip(preds, image_names):
        data[image_name] = {str(ind + 1): round(float(val), 2) for ind, val in enumerate(pred)}  # np float=>>float(val)
    with open(os.path.join(config.SUBMIT_PATH, 'first_round_test_submit.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print("write result success")


def inference(args):
    # build model
    batch_size = args.batch_size

    model = torchvision.models.resnet18(pretrained=False, num_classes=len(config.labels))
    model = model.to(device)

    # load model
    best_model_file = os.listdir(config.BEST_MODEL_PATH)[-1]  # 默认取最近时间的一个
    print("Loading model from ", best_model_file)
    model.load_state_dict(torch.load(os.path.join(config.BEST_MODEL_PATH, best_model_file)))
    model.eval()
    # load data
    test_loader = DataLoader(TestDataset(), batch_size=batch_size)

    preds = []
    image_names = []

    # inference
    for images, image_name in test_loader:
        images = images.to(device)
        outputs = model(images)
        pred = torch.sigmoid(outputs)
        preds.append(pred.cpu().detach().numpy())
        image_names.append(image_name)

    preds = np.concatenate(preds, axis=0)
    image_names = np.concatenate(image_names, axis=0)
    # save final result
    write_submit_result(preds, image_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tianchi Jinnan ')

    parser.add_argument('--batch_size', type=int, required=True, default=8)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    create_logging("../logs", "w")
    logging.info(args)
    inference(args)
