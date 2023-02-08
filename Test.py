import csv
from os import path

import cv2
import torch

basePath = "Dataset_2"

if __name__ == '__main__':

    model = torch.load('model.mdl')

    model.eval()

    with open(path.join(basePath, "Train.csv")) as f:
        lines = csv.reader(f)
        title = True
        for line in lines:
            if title:
                title = False
                continue
            img = cv2.imread(path.join(basePath, line[7]))
            x = torch.tensor(img, dtype=torch.float)
            x = x.ravel()
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            print(predicted)




