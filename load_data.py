from config import FLAGS as cfg
import numpy as np
import random
import cv2
import os

class data(object):
    def __init__(self):
        self.train_data = get_data_list(cfg.train_data) # 964
        self.test_data = get_data_list(cfg.test_data) # 659
        self.data = {"train":self.train_data, "test":self.test_data}
        self.len  = {"train":len(self.train_data), "test":len(self.test_data)}
        self.count = {"train":0, "test":0}
        # cfg.ways*cfg.shot 一个batch中memory有ways*shot个图片
        self.memory_x = np.zeros((cfg.batch_size, cfg.ways*cfg.shot, cfg.input_image[0], cfg.input_image[1], cfg.input_image[2]))  # memory
        self.memory_label = np.zeros((cfg.batch_size,  cfg.ways*cfg.shot), dtype=np.int)
        self.classify_x = np.zeros((cfg.batch_size, cfg.input_image[0], cfg.input_image[1], cfg.input_image[2])) # target
        self.classify_label = np.zeros((cfg.batch_size,), dtype=np.int)
        self.shuffle = True  # shuffle data after one epoch


    def __call__(self, model="train"):
        # if (self.count[model] + 1)*cfg.batch_size > self.len[model]:
        #     self.count[model] = 0
        #     if self.shuffle:
        #         random.shuffle(self.data[model])

        for i in range(cfg.batch_size):  # 每一个batch是独立的
            ind = 0
            pinds = np.random.permutation(cfg.ways*cfg.shot)  # 0到25之间随机取数  memory有ways*shot张图片
            # 比如 当ways=5,shot=5时 pinds [20  1  0  8 11 13 15 21 24 18  9  5 22  2 10 16 17 19  3 12 23  6  7 14  4]
            classes = np.random.choice(self.len[model], cfg.ways, False)  # 数据集中随机选5个类别作为memory
            # class [ 318 900   42  524  759]
            x_hat_class = np.random.randint(cfg.ways)  # 选取哪一类作为target分类 0 ～ ways-1
            # x_hat_class  4

            # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            for j, cur_class in enumerate(classes):  # 对每一类
                example_inds = np.random.choice(len(self.data[model][cur_class]), cfg.shot, False)  # 每组数据中有20个 我们随机取shot个作为memory
                # example_inds [12 15  2  4 18]

                for eind in example_inds:  # example_inds一类类别取shot个len(example_inds)=cfg.shot
                    self.memory_x[i, pinds[ind], ...] = data_argument(self.data[model][cur_class][eind])[:, :, :cfg.input_image[2]]
                    self.memory_label[i, pinds[ind]] = j  # 记录每一个的label
                    ind += 1
                if j == x_hat_class:  # 第4类作为要分类的数据
                    # print("hat", x_hat_class, self.data[model][cur_class][eind])
                    # 在当前cur_class类下随机取一个  可能选择的有可能和memory是一张图片
                    target_index = np.random.choice(20)
                    self.classify_x[i, ...] =  data_argument(self.data[model][cur_class][target_index])[:, :, :cfg.input_image[2]]
                    # print(target_index, self.data[model][cur_class][target_index])
                    # input()
                    self.classify_label[i] = j  # 4
        return (self.memory_x - cfg.data_mean)/cfg.data_std, self.memory_label, (self.classify_x - cfg.data_mean)/cfg.data_std, self.classify_label

def data_argument(name):
    # print("name", name)
    im = cv2.imread(name)
    im = np.rot90(im, np.random.randint(4))  # 图片随机旋转
    im = cv2.resize(im, (cfg.input_image[0], cfg.input_image[1]))
    return im

def get_data_list(path):
    data_list = []
    m_l = []
    s_l = []
    for child in os.listdir(path):
        for character in os.listdir(os.path.join(path, child)):
            one_cls = []   # 一个类别20张图片
            for png in os.listdir(os.path.join(path, child, character)):
                name = os.path.join(path, child, character, png)
                im = cv2.imread(name)
                one_cls.append(name)
            data_list.append(one_cls)
    return data_list
# ------------------------------------------------------------------------------
