import re
import matplotlib.pyplot as plt
import os
import numpy as np


def load_log_file(path):
    """
    正则匹配出所有相关数据并返回
    :param path:
    :return:
    """
    with open(path, mode='r') as file:
        train_loss, test_loss, train_map, test_map, train_mauc, test_mauc = [], [], [], [], [], []

        for line in file.readlines():
            line = line.strip()
            pattern1 = re.compile(r'train_loss: ([\d.]+) train_mAP: ([\d.]+) train_mAUC: ([\d.]+)')
            for item in re.findall(pattern1, line):
                if len(item):
                    train_loss.append(float(item[0]))
                    train_map.append(float(item[1]))
                    train_mauc.append(float(item[2]))

            pattern2 = re.compile(r'val_loss: ([\d.]+) val_mAP: ([\d.]+) val_mAUC: ([\d.]+)')
            for item in re.findall(pattern2, line):
                test_loss.append(float(item[0]))
                test_map.append(float(item[1]))
                test_mauc.append(float(item[2]))
    return train_loss, train_map, test_mauc, test_loss, test_map, test_mauc


def plot_experiment(log_path):
    train_loss, train_map, train_mauc, test_loss, test_map, test_mauc = load_log_file(log_path)

    epochs = range(len(train_loss))
    max_auc = np.max(test_mauc)
    max_auc_epoch = epochs[np.argmax(test_mauc)]
    print(max_auc)
    print(max_auc_epoch)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplot(131)
    ## 标注最大值的点
    plt.scatter([max_auc_epoch, ], [max_auc, ], s=50, color='b', zorder=1)
    ##注释
    plt.annotate('({},{})'.format(max_auc_epoch, max_auc), xy=(max_auc_epoch, max_auc), xycoords='data',
                 xytext=(-60, -60),
                 textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    ## 垂线
    # plt.plot([max_acc_iter, max_acc_iter, ], [0, max_acc, ], 'k--', linewidth=2.5)
    p1 = plt.plot(epochs, train_mauc, '.--', color='#6495ED', zorder=0)
    p2 = plt.plot(epochs, test_mauc, '.--', color='#FF6347', zorder=0)
    plt.legend([p1[0], p2[0]], ['train_mauc', 'test_mauc'])
    plt.title('train/test mAUC')
    plt.xlabel('Epoch')
    plt.ylabel('mAUC')
    plt.ylim((0, 1))
    # plt.xscale('log')

    plt.subplot(132)
    p1 = plt.plot(epochs, train_map, '.--', color='#6495ED')
    p2 = plt.plot(epochs, test_map, '.--', color='#FF6347')
    plt.legend([p1[0], p2[0]], ['train_map', 'test_map'])
    plt.title('train/test mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.ylim((0, 1))
    # plt.xscale('log')

    plt.subplot(133)
    p1 = plt.plot(epochs, train_loss, '.--', color='#6495ED')
    p2 = plt.plot(epochs, test_loss, '.--', color='#FF6347')
    plt.legend([p1[0], p2[0]], ['train_loss', 'test_loss'])
    plt.title('train/test loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    # plt.xscale('log')

    fig.tight_layout()  ## 这个很好用 可以防止规范子图的显示范围
    plt.show()


def show_log_curve(index):
    workspace = "../../logs"
    log_path = os.path.join(workspace, '{}.log'.format(index))
    plot_experiment(log_path)


if __name__ == '__main__':
    show_log_curve('0024')
