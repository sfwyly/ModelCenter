
"""
    @Author: Junjie Jin
    @Date: 2021/6/9
    @Description: 训练中心

"""
import numpy as np


class Trainer:

    def __init__(self):
        pass

    @classmethod
    def preprocess(data):
        """
        数据预处理
        :param data: 待处理的数据
        :return: 处理好的数据
        """

        ds = data.shape

        assert len(ds) >= 2, "数据宽度不能小于2"

        params = [[],[],[]]

        for _ in range(ds[-1]):

            params[0].append(np.mean(data[..., _]))
            params[1].append(np.max(data[..., _]))
            params[2].append(np.min(data[..., _]))

        params = np.array(params)
        return (data - params[0])/ (params[1] - params[2]), params

    def train(self, model, data, epochs, valid_per_epoch, batch_size):

        """
        训练执行 传递忙获取到的模型与处理好数据 进行训练
        data 同时具有的是数据与标签与其他数据类型 然而
        :param model: 训练模型
        :param epochs: 训练
        :param valid_per_epoch:验证批次
        :param batch_size: 批量大小
        :return: 返回结果
        """
        data, params = self.preprocess(data)
        for i in range(epochs):

            for i,(x, y) in enumerate(dataloader):

                y_pred = model(x)
                loss = model.cal_loss(y, y_pred)




    def swap(self):
        """
        训练实时数据进行传递到前端
        :return:
        """

        return

    def call(self):

        pass




