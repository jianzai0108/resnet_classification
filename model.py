import torch
import torch.nn as nn
import torchvision.models as models


def my_resnet(output_dim):
    pretrained_model = models.resnet50(pretrained=True)

    IN_FEATURES = pretrained_model.fc.in_features
    OUTPUT_DIM = output_dim

    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    pretrained_model.fc = fc

    # print(pretrained_model.fc)

    return pretrained_model