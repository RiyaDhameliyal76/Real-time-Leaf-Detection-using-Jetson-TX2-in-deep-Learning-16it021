import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import scipy.misc
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torchvision.models as models
import utils

from PIL import Image
from averagemeter import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms

model = models.resnet101(pretrained=True)
model.fc = nn.Linear(2048, 185)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
min_img_size = 224
transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
traindir = os.path.join('dataset', 'train')
data_train = datasets.ImageFolder(traindir, transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]))
classes=data_train.classes
model.eval()
stream=cv2.VideoCapture(0)
while True:
    frame1=stream.read()
    frame=transform_pipeline(frame1)
    output = model(frame)
    cv2.imshow('frame',frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(output)
