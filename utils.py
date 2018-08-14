from config import FLAGS as cfg
import matplotlib.pyplot as plt
import random
import csv

def lr(count):
    if count < 101:
        cfg.learning_rate = 0.0001
    elif 100 < count < 1000:
        cfg.learning_rate = 0.001
    else:
        cfg.learning_rate = 0.0001

def save_csv(name, contexts):
    """
    :param name:  要保存的csv文件名， 如：path/to/gps.csv或 .txt也可以
    :param context:  要保存的文本信息，是一个列表形式：
    gps = [('latitude', 'longitude'), (30.745143, 103.927407), (30.746547, 103.928599), (30.749746, 103.931295),...
           第一行表示名称，后面是值
    :return: 返回一个新的csv文件
    """
    with open(name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for context in contexts:
            csv_writer.writerow(context)

def read_csv(csv_name):
    context = csv.reader(open(csv_name, 'r'))
    cont_list = []
    for file in context:
        cont_list.append(file)
    return cont_list


def plot_learning_curves(fig_path,_axis, data_list, name, title,  color_list=None, _eps=True,_len_axis=5, xlabel='iter', ylabel='Acc', linestyle='-',
                         marker=None, style=''):
    """
    :param fig_path: 图片保存位置（包括文件名）
    :param _axis: 坐标标注
    :param n_epochs: 数据长度
    :param data_list: 数据列表，可以是多个数据，但是每个数据长度必须一样, [[数据集1], [数据集2]， ...]
    :param name: 每个数据的名称 也是一个列表
    :param color_list: 每个数据对应颜色列表,可以选：['dodgerblue', 'red', 'aqua', 'orange']
    :param title: 图形标题
    :param xlabel: x坐标名称
    :param ylabel：y坐标名称
    :param xticks: [[...],[...]] 两个列表分别用于显示在哪个位置
    :param style: ...
    :param _len_axis: 坐标轴显示几个横坐标
    :param _eps: 输入数据第一列index是eps，还是iter
    :param linestyle: 画的线的格式 可以是直线 ‘-’  虚线： ‘--’
    :param marker:    每个点标记 可以是圆圈： ‘o’，默认是None
    :return:
    """
    if color_list is None:
        color_list = ['dodgerblue', 'red', 'aqua', 'orange']
    measure = ylabel
    steps_measure = xlabel

    plt.figure(dpi=400)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    n_epochs = len(data_list[0])
    steps = range(1, n_epochs + 1)
    plt.title(title + style)
    for ii in range(len(data_list)):
        assert ii < len(color_list)
        plt.plot(steps, data_list[ii], linewidth=1, color=color_list[ii], linestyle=linestyle, marker=marker,
                 markeredgecolor='black',
                 markeredgewidth=0.5, label=name[ii])
    eps = int((steps[-1] - 1) / _len_axis)
    if _eps:
         # print(_axis[eps], _axis[eps * 2], _axis[eps * 3], _axis[eps * 4],_axis[eps * 5])
         plt.xticks([0, eps, eps * 2, eps * 3, eps * 4, eps * 5],  # x轴刻度
             [0,_axis[eps], _axis[eps * 2], _axis[eps * 3], _axis[eps * 4], _axis[eps * 5]]
               )  # 前面一个数组表示真真实的值，后面一个表示在真实值处显示的值
    else:   # 输入是iter迭代次数
         plt.xticks([0, eps, eps * 2, eps * 3, eps * 4, eps * 5],
               [0, eps, eps * 2, eps * 3, eps * 4, eps * 5])  # 前面一个数组表示真真实的值，后面一个表示在真实值处显示的值

    plt.xlabel(steps_measure)
    plt.ylabel(measure)
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(fig_path)  # 这一句要在plt.show()之前
    plt.show()

def plot_scatter(fig_path, data_list, label_list, title, color_list=['dodgerblue', 'aqua', 'red', 'orange'], xlabel='iter', ylabel='Acc', style='', alpha=0.3):
    """
    :param fig_path: 图片保存位置（包括文件名）
    :param data_list: 数据列表，可以是多个数据，但是每个数据长度必须一样, [[数据集1], [数据集2]， ...]
    :param label_list: 每个数据的名称 也是一个列表
    :param color_list: 每个数据对应颜色列表,可以选：['dodgerblue', 'red', 'aqua', 'orange']
    :param title: 图形标题
    :param xlabel: x坐标名称
    :param ylabel：y坐标名称
    :param style: ...
    :return:
    """
    fig, ax = plt.subplots()
    steps = range(1, len(data_list[0]) + 1)

    for index, data in enumerate(data_list):
        scale = 200.0 * random.random()
        ax.scatter(steps, data, c=color_list[index], s=scale, label=label_list[index],
                   alpha=alpha, edgecolors='none')

    ax.legend()
    ax.grid(True)
    plt.title(title + style)
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(fig_path)  # 这一句要在plt.show()之前

    plt.show()
