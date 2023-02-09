import datetime

import torch
from torch.utils.data import DataLoader
import pandas as pd
from MyDataset import TrainingDataset, TestingDataset
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


def plot_performance(history=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    # legends = ['Training']

    plt.figure(figsize=(20, 5))

    y1 = history['AccAvg']

    min_y = min(y1) - ylim_pad[0]
    max_y = max(y1) + ylim_pad[0]

    plt.subplot(121)

    plt.plot(y1)

    # plt.title('Model Training Accuracy\n' + date_time(1), fontsize=17)
    plt.title('Model Training Accuracy', fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    # plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history['LossAvg']

    min_y = min(y1) - ylim_pad[1]
    max_y = max(y1) + ylim_pad[1]
    plt.subplot(122)

    plt.plot(y1)

    # plt.title('Model Training Loss\n' + date_time(1), fontsize=17)
    plt.title('Model Training Loss', fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    # plt.legend(legends, loc='upper left')
    plt.grid()

    plt.savefig("history.png")

    plt.show()


if __name__ == '__main__':

    lr = 0.005
    batch_size = 1024
    epoches = 20

    LossAvg = []
    AccAvg = []
    TestAccAvg = []

    train_dataset = TrainingDataset()
    test_dataset = TestingDataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

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
        LossAvg.append(sum_loss / len(train_dataloader))
        AccAvg.append(sum_acc / len(train_dataloader))
        print('avg_loss:', sum_loss / len(train_dataloader), 'acc:', sum_acc / len(train_dataloader))

        model.eval()
        path = []
        truth = []
        act = []
        sum_test_acc = 0.
        for hist, label in test_dataloader:
            hist = hist.cpu()
            label = label.cpu()
            y = model(hist)

            acc = 0
            for i in range(y.shape[0]):
                index = y[i].argmax().item()
                if index == label[i].argmax().item():
                    acc += 1
                truth.append(label[i].argmax().item())
                act.append(y[i].argmax().item())
                path.append(test_dataset.data_path[i])
            acc /= y.shape[0]
            sum_test_acc += acc
        TestAccAvg.append(sum_test_acc / len(test_dataloader))
        print('test_acc', sum_test_acc / len(test_dataloader))
        epochDf = {'path': path, 'truth': truth, 'act': act}
        epochDf = pd.DataFrame(epochDf)
        epochDf.to_csv(f'out/epoch_{epoch + 1}.csv')

    df = {'LossAvg': LossAvg, 'AccAvg': AccAvg, "TestAcc": TestAccAvg}
    df = pd.DataFrame(df)
    df.to_csv(f'lr_{lr}_bs_{batch_size}_epoch_{epoches}.csv')
