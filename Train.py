import torch
from torch.utils.data import DataLoader
import pandas as pd
from MyDataset import MyDataset
from Model import Model

if __name__ == '__main__':

    lr = 0.005
    batch_size = 1024
    epoches = 32

    Loss = []
    Acc = []

    train_dataset = MyDataset()

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
        print('avg_loss:', sum_loss / len(train_dataloader), 'acc:', sum_acc / len(train_dataloader))

    df = {'Loss': Loss, 'Acc': Acc}
    df = pd.DataFrame(df)
    df.to_csv(f'lr_{lr}_bs_{batch_size}_epoch_{epoches}.csv')
    torch.save(model, 'model.mdl')
