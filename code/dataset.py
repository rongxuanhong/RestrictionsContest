import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import json
import config
import random

# 1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


def parse_json(json_path):
    """解析限制品的数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
        labels = np.zeros((len(data["images"]), len(config.labels)))

        images = [(image["file_name"], 1) for image in data["images"]]

        print("总共有{}张限制品图片".format(len(images)))
        for ind, annotation in enumerate(data["annotations"]):
            image_id = annotation["image_id"]
            category_id = annotation["category_id"] - 1
            labels[image_id][category_id] = 1

        assert len(images) == len(labels)
        return np.array(images), labels


def load_normal_images():
    images = [(image.split('/')[-1], 0) for image in os.listdir(config.NORMAL_PATH)]

    labels = np.tile([0, 0, 0, 0, 0, 1], (len(images), 1))
    assert len(images) == len(labels)
    return np.array(images), labels


class TrainDataset(Dataset):
    def __init__(self, json_path, transform=None):
        ##1.加载限制品数据
        self.images, self.labels = parse_json(json_path)
        # ## 2.加载正常数据
        # images_norm, labels_norm = load_normal_images()
        # self.images = np.concatenate((images_res, images_norm), axis=0)
        # self.labels = np.concatenate((labels_res, labels_norm), axis=0)

        print("Loading {} images to train".format(len(self.images)))

        if transform:
            self.composed = transform
        else:
            self.composed = transforms.Compose([
                transforms.Resize((config.img_height, config.img_weight)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index):
        file_name, mark = self.images[index]
        if mark:
            img = Image.open(os.path.join(config.RESTRICTED_PATH, file_name))
        else:
            img = Image.open(os.path.join(config.NORMAL_PATH, file_name))
        img = self.composed(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.images)

    # def collate_fn(batch):
    #     imgs = []
    #     label = []
    #     for sample in batch:
    #         imgs.append(sample[0])
    #         label.append(sample[1])
    #
    #     return torch.stack(imgs, 0), label


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.images = [(image.split('/')[-1], 0) for image in os.listdir(config.TEST_PATH)]
        if transform:
            self.composed = transform
        else:
            self.composed = transforms.Compose([
                transforms.Resize((config.img_height, config.img_weight)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name,mark = self.images[index]
        img = Image.open(os.path.join(config.TEST_PATH, image_name))
        img = self.composed(img)
        return img, image_name


def load_split_train_val(val_portion=.2):
    train_dataset = TrainDataset(config.TRAIN_NO_POLY_JSON_PATH, )
    val_dataset = TrainDataset(config.TRAIN_NO_POLY_JSON_PATH, )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_portion * num_train))

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)

    return {'train': train_loader, 'val': val_loader}


if __name__ == '__main__':
    # loader = load_split_train_val(val_portion=0.2)
    # print(len(loader[0]) * 8)
    data=TestDataset()
    for a in data:
        print(a)
