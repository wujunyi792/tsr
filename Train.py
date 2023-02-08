import datetime

import torch
from torch.utils.data import DataLoader
import pandas as pd
from MyDataset import TrainingDataset
from Model import Model
import matplotlib.pyplot as plt


def date_time(x):
    if x == 1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 2:
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 3:
        return 'Date now: %s' % datetime.datetime.now()
    if x == 4:
        return 'Date today: %s' % datetime.date.today()


def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history['AccAvg']
    y2 = history['AccAvg']

    min_y = min(min(y1), min(y2)) - ylim_pad[0]
    max_y = max(max(y1), max(y2)) + ylim_pad[0]

    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n' + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history['LossAvg']
    y2 = history['LossAvg']

    min_y = min(min(y1), min(y2)) - ylim_pad[1]
    max_y = max(max(y1), max(y2)) + ylim_pad[1]
    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n' + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory + "/history")

    plt.show()


if __name__ == '__main__':

    lr = 0.005
    batch_size = 1024
    epoches = 32

    Loss = []
    Acc = []
    LossAvg = []
    AccAvg = []
    Epoch = []

    train_dataset = TrainingDataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Model().cpu()
    # model = torch.load('model.mdl')
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = torch.nn.CrossEntropyLoss().cpu()
    for epoch in range(epoches):
        print('epoch:', epoch + 1)
        sum_loss = 0.
        sum_acc = 0.
        model.train()
        for hist, label in train_dataloader:
            # print(hist)
            hist = hist.cpu()
            label = label.cpu()

            y = model(hist)

            acc = 0
            for i in range(y.shape[0]):
                index = y[i].argmax().item()
                if index == label[i].argmax().item():
                    acc += 1
                # print(index,label[i].argmax().item())
            acc /= y.shape[0]
            # print(acc)

            loss = loss_fn(y, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sum_loss += loss.item()
            sum_acc += acc
            # print(loss.item())
            Loss.append(loss.item())
            Acc.append(acc)
        LossAvg.append(sum_loss / len(train_dataloader))
        AccAvg.append(sum_acc / len(train_dataloader))
        Epoch.append(epoch + 1)
        print('avg_loss:', sum_loss / len(train_dataloader), 'acc:', sum_acc / len(train_dataloader))

    df = {'Loss': Loss, 'Acc': Acc}
    df = pd.DataFrame(df)
    df.to_csv(f'lr_{lr}_bs_{batch_size}_epoch_{epoches}.csv')
    torch.save(model, 'model.mdl')
    plot_performance(history={"LossAvg": LossAvg, "AccAvg": AccAvg, "Epoch": Epoch})
