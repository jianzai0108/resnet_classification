import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from model import my_resnet

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm, trange

# from sklearn import decomposition
# from sklearn import manifold
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time
from data_preprocess import make_train_test

SEED = 1234
TRAIN_RATIO = 0.8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def calculate_topk_accuracy(y_pred, y, k=1):
    # top_pred = y_pred.argmax(1, keepdim=True)
    # correct = top_pred.eq(y.view_as(top_pred)).sum()
    # acc = correct.float() / y.shape[0]
    # return acc
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k


def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_5


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_5


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(model, EPOCHS, train_iterator, valid_iterator, optimizer, criterion, scheduler, device):
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.monotonic()

        train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, scheduler, device)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            path = 'save_model_' + str(time.asctime()) + '.pt'
            torch.save(model.state_dict(), path)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | ' \
              f'Train Acc @5: {train_acc_5 * 100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | ' \
              f'Valid Acc @5: {valid_acc_5 * 100:6.2f}%')


def get_predictions(model, iterator):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs


if __name__ == '__main__':

    image_dir = './dataset/images'
    train_dir = './dataset/train'
    test_dir = './dataset/test'

    print('Data preparing...')
    train_data, valid_data, test_data = make_train_test(image_dir, train_dir, test_dir, flag=True)
    print('Data prepare finished.')

    BATCH_SIZE = 64

    train_iterator = data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
    valid_iterator = data.DataLoader(valid_data, BATCH_SIZE)
    test_iterator = data.DataLoader(test_data, BATCH_SIZE)

    EPOCHS = 20
    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

    START_LR = 1e-3

    print('Model building...')
    model = my_resnet(6)
    optimizer = optim.Adam(model.parameters(), lr=START_LR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # MAX_LRS = [p['lr'] for p in optimizer.param_groups]

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8, last_epoch=-1)
    print('Model build finished.')

    print('Start training...')
    train_model(model, EPOCHS, train_iterator, valid_iterator, optimizer, criterion, scheduler, device)
    print('Training finished.')
    # test
    # model.load_state_dict(torch.load('tut5-model2.pt'))
    #
    # test_loss, test_acc_1, test_acc_5 = evaluate(model, test_iterator, criterion, device)
    #
    # print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1 * 100:6.2f}% | ' \
    #       f'Test Acc @5: {test_acc_5 * 100:6.2f}%')