import os
import random
import torch
import torchvision
import numpy as np
import warnings
from torch import nn, optim
from torch.optim import lr_scheduler
import config
import time
from dataset import load_split_train_val
from utils.utils import *
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import argparse
from models.darknet import darknet

# 1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, args, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = .0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}.'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = .0
            y_preds = []
            y_scores = []
            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)
                y_scores.append(labels)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs.type(torch.double), labels.type(torch.double))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs)
                y_preds.append(preds)
                # predict = (preds > 0.5).float()

            y_preds = torch.cat(y_preds)
            y_trues = torch.cat(y_scores)

            # y_preds = (y_preds > 0.5).float() * y_preds
            roc_auc = roc_auc_score(y_trues.detach().cpu().numpy(), y_preds.detach().cpu().numpy())
            mAP = average_precision_score(y_trues.detach().cpu().numpy(), y_preds.detach().cpu().numpy())

            total = len(dataloaders[phase]) * args.batch_size
            epoch_loss = running_loss / total

            if phase == "train":
                logging.info('train_loss: {:.4f} train_mAP: {:.4f} train_mAUC: {:.4f}'.format(epoch_loss, mAP, roc_auc))
            else:
                logging.info('val_loss: {:.4f} val_mAP: {:.4f} val_mAUC: {:.4f}'.format(epoch_loss, mAP, roc_auc))

            if phase == 'val' and mAP > best_acc:
                best_acc = mAP
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    logging.info('Training Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(),
               os.path.join('best_model', 'best_model_' + str(datetime.now()) + '.pt'))
    return model


# def visualize_model(model, dataloaders, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(dataloaders['val']):
#             images = images.to(device)
#             outputs = model(images)
#             preds = torch.sigmoid(outputs)
#             preds = torch.nonzero((preds > 0.5).float() * outputs).squeeze()
#
#             for j in range(images.size(0)):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, images_so_far)
#                 ax.axis('off')
#                 str = ''
#                 preds = preds.cpu().numpy()
#                 for index, val in enumerate(preds):
#                     str += config.ind_to_label[val]
#                     if index < len(preds) - 1:
#                         str += ","
#                 ax.set_title("predicted:{}".format(str))
#                 plt.imshow(images.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.training(was_training)
#                     return
#         model.training(was_training)


def train(args):
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs

    # model = torchvision.models.resnet18(pretrained=False, num_classes=len(config.labels))

    model = darknet(num_classes=5)

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer_fit = optim.Adam(model.parameters(), lr=lr, )

    exp_lr_schedule = lr_scheduler.StepLR(optimizer_fit, step_size=1, gamma=0.95)

    dataloaders = load_split_train_val(batch_size)

    model_fit = train_model(model, args, dataloaders, criterion, optimizer_fit, exp_lr_schedule, num_epochs=epochs)

    # visualize_model(model_fit, dataloaders=dataloaders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tianchi Jinnan ')

    parser.add_argument('--batch_size', type=int, required=True, default=8)
    parser.add_argument('--learning_rate', type=float, required=True, default=1e-3)
    parser.add_argument('--epochs', type=int, required=True, default=20)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # # Create log
    create_logging("../logs", "w")
    logging.info(args)

    train(args)
