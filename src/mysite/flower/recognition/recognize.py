import torch
import numpy as np
from torchvision import transforms
import cv2

from flower.recognition.model import *


def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u


def main(img_path):
    model_path = 'flower/recognition/save_model/model_20.pth'
    label = ['Tulip',
             'Snowdrop',
             'LilyValley',
             'Bluebell',
             'Crocus',
             'Iris',
             'Tigerlily',
             'Daffodil',
             'Fritillary',
             'Sunflower',
             'Daisy',
             'ColtsFoot',
             'Dandelion',
             'Cowslip',
             'Buttercup',
             'Windflower',
             'Pansy']
