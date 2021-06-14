

"""
    @Author: Junjie Jin
    @Date: 2021/6/9
    @Description: 均方错误损失

"""

import torch
import numpy as np


class MSE:

    def __init__(self, is_torch: bool):
        """
        :param is_torch: 是否是使用了torch框架
        """

        self.is_torch = is_torch

    def call(self, x_true, x_predict):

        if self.is_torch:

            return torch.mean((x_true - x_predict)**2)

        return np.mean((x_true - x_predict)**2)