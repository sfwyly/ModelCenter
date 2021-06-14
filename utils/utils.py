

from losses import *

"""
    损失类中心字段
"""
Loss_Factory = {

    "MSE": MSE
}



"""
    根据字符串的列表返回对应的loss 类列表
"""


def getLossClass(loss_list: list):

    loss_class = []

    for s in loss_list:

        loss = getClass(s)
        loss_class.append(loss)

    return loss_class


def getClass(s):

    return Loss_Factory[s]