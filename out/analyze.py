import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def read_label():
    res = []
    for i in range(44):
        res.append(i)
    return res


if __name__ == '__main__':
    for i in range(1, 21):
        arr = np.zeros((43, 43))
        with open(f"epoch_{i}.csv") as f:
            f = csv.reader(f)
            f.__next__()
            for line in f:
                arr[int(line[2])][int(line[3])] += 1

        plot_confusion_matrix(arr, read_label(), f"Confusion Matrix On Test Dataset(Epoch {i})")
        # plt.savefig(f"Confusion Matrix On Test Dataset(Epoch {i}).png", format='png')
        plt.show()
