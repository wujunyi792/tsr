import csv
from os import path

import cv2
import torch

basePath = "Dataset_2"

if __name__ == '__main__':

    model = torch.load('model.mdl')
