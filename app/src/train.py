#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
# @Time    : 8/4/2021 12:14 PM
# @Author  : sanket.tijare@icertis.com
# @File    : train.py
"""
import pandas as pd
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob

BOOTS_DATA_PATH = '../../data/boots/'
SANDALS_DATA_PATH = '../../data/sandals/'
SHOES_DATA_PATH = '../../data/shoes/'
SLIPPERS_DATA_PATH = '../../data/slippers/'

DB_PATH = '../db/'


class SetSeed:
    def setter(self):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG = models.vgg19(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output


def extractor(img_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    return y


def main(directories):
    SS = SetSeed()
    SS.setter()
    model = Encoder()

    for data_dir in directories:
        print(f"Current SKU Type - {data_dir}")
        files_list = []
        x = os.walk(data_dir)
        for path, d, filelist in x:
            for filename in filelist:
                file_glob = os.path.join(path, filename)
                files_list.extend(glob.glob(file_glob))

        filename = []
        features = []
        for x_path in files_list:
            # print("x_path" + x_path)
            file_name = x_path.split('/')[-1]
            bottleneck_features = extractor(x_path, model, False)
            filename.append(x_path)
            features.append(list(bottleneck_features.reshape(1, -1)))

        data = pd.DataFrame({"filename": filename, "features": features})
        save_path = os.path.join(DB_PATH, data_dir.split("/")[-2] + ".pkl")
        data.to_pickle(save_path)


if __name__ == "__main__":
    sku_list = [BOOTS_DATA_PATH, SANDALS_DATA_PATH, SHOES_DATA_PATH, SLIPPERS_DATA_PATH]
    main(sku_list)
