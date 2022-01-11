import os
import shutil
import copy
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

RANDOM_SEED = 1994
random.seed(RANDOM_SEED)


def make_train_test(images_dir, train_dir, test_dir, flag=False, TRAIN_RATIO=0.8, VALID_RATIO=0.9):
    if flag:
        classes = os.listdir(images_dir)

        for c in classes:

            class_dir = os.path.join(images_dir, c)
            images = os.listdir(class_dir)

            # 随机打乱图片顺序
            random.shuffle(image)

            n_train = int(len(images) * TRAIN_RATIO)

            train_images = images[:n_train]
            test_images = images[n_train:]

            os.makedirs(os.path.join(train_dir, c), exist_ok=True)
            os.makedirs(os.path.join(test_dir, c), exist_ok=True)

            for image in train_images:
                image_src = os.path.join(class_dir, image)
                image_dst = os.path.join(train_dir, c, image)
                shutil.copyfile(image_src, image_dst)

            for image in test_images:
                image_src = os.path.join(class_dir, image)
                image_dst = os.path.join(test_dir, c, image)
                shutil.copyfile(image_src, image_dst)

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(pretrained_size, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds)
    ])

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transforms)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transforms)

    n_train_examples = int(len(train_data))
    # n_train_examples = int(len(train_data) * VALID_RATIO)
    # n_valid_examples = len(train_data) - n_train_examples

    # train_data, valid_data = data.random_split(train_data,
    #                                            [n_train_examples, n_valid_examples])

    # valid_data = copy.deepcopy(valid_data)
    # valid_data.dataset.transform = test_transforms
    print(f'Number of training examples: {len(train_data)}')
    # print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    return train_data, test_data
    # return train_data, valid_data, test_data