import numpy as np
import cv2

class load_data_minibatch():
    def __init__(self, data, batch_size, memory_size, input_image, ways, shot, reuse=False):
        self.cur_data = data
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.input_image = input_image
        self.ways = ways
        self.shot = shot
        self.reuse = reuse    # 默认不重复使用变量名称，使所有变量名不一样
        # reuse:是否重新使用当前变量名    reuse: Boolean or None, setting the reuse in get_variable.
        # name: name of the current scope, used as prefix in get_variable.
        # tensorflow 为了更好的管理变量,提供了variable scope机制
    def get_minibatch(self):
        memory_x = np.zeros((self.batch_size, self.memory_size, self.input_image[0], self.input_image[1], 1))
        memory_label = np.zeros((self.batch_size, self.memory_size), dtype=np.int)
        classify_x = np.zeros((self.batch_size, self.input_image[0], self.input_image[1], 1))
        classify_label = np.zeros((self.batch_size,), dtype=np.int)
        for i in range(self.batch_size):  # 每一个batch是独立的
            ind = 0
            pinds = np.random.permutation(self.memory_size)  # 0到25之间随机取数
            # pinds [20  1  0  8 11 13 15 21 24 18  9  5 22  2 10 16 17 19  3 12 23  6  7 14  4]
            # print("pinds", pinds)
            classes = np.random.choice(self.cur_data.shape[0], self.ways, False)  # 0 1200随机选5组数据
            # class [ 318 1150   42  524  759]
            # print("class", classes)
            x_hat_class = np.random.randint(self.ways)  # 0 4  抽取一个类作为要classify的数据输入label
            # x_hat_class  4
            # print("x_hat_class", x_hat_class)

            # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            for j, cur_class in enumerate(classes):  # each class  j是索引0到4中某一类 cur_class是取的某一组
                # print("j", j, cur_class)
                # 第一个j cur_class 0 318  （共5组）  j是第几类
                example_inds = np.random.choice(self.cur_data.shape[1], self.shot, False)  # 每组数据中有20个 我们随机取5个
                # example_inds [12 15  2  4 18]
                # print("example_inds", example_inds)
                for eind in example_inds:  # example_inds这里只有一个数  每一类取一个
                    # print("eind", eind)
                    memory_x[i, pinds[ind], :, :, 0] = np.rot90(self.cur_data[cur_class][eind],
                                                                np.random.randint(4))  # 图片随机旋转
                    # memory_x[0,20,:,:,0] = self.cur_data[318,12,:,:]
                    # memory_x[0,1,:,:,0] = self.cur_data[318,15,:,:]
                    memory_label[i, pinds[ind]] = j
                    # memory_label[0,20] = 0
                    # memory_label[0,1] = 0
                    ind += 1
                if j == x_hat_class:  # 第4类作为要分类的数据    在当前cur_class类下随机取一个
                    classify_x[i, :, :, 0] = np.rot90(self.cur_data[cur_class][np.random.choice(self.cur_data.shape[1])],
                                                      np.random.randint(4))
                    # cv2.imshow("target", classify_x[i, :, :, 0])
                    # cv2.waitKey()
                    # print(cur_class, j)
                    classify_label[i] = j  # 4
        return memory_x, memory_label, classify_x, classify_label
        # support set: memory_x memory_label    classifier set: classify_x classify_label