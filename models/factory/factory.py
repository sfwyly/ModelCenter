
"""
    @Author: Junjie Jin
    @Date: 2021/6/9
    @Description: 模型工厂

"""

from models import *


class ClassPool:

    def __init__(self):
        # todo 自定义读取动态加载
        # 模型字典 str->model_class
        self.classes = {
            "MLP": MLP
        }

    def getModelClass(self, class_name: str):
        """
        :param class_name:
        :return: 返回类名复用
        """

        return self.classes[class_name]


class KeyMap:

    def __init__(self, name, kwargs):

        """
        映射每个模型，支持复用
        :param name:
        :param kwargs:参数字典
        """

        self.name = name
        self.kwargs = kwargs

    def __hash__(self):

        return hash(self.name)+hash(str(self.kwargs))

    def __eq__(self, other):

        return (self.name, self.kwargs) == (other.name, other.kwargs)


class ModelPool:

    def __init__(self, classpool: ClassPool):
        """
        模型池
        :param classPool:
        """
        self.models = dict()

        self.classpool = classpool

    def getModel(self, name, args):
        """
        获取对应买模型
        :param name: 模型名
        :param args: 模型参数字典
        :return:
        """
        keymap = KeyMap(name, args)

        if keymap in self.models:
            return self.models[keymap]

        # 不存在直接调用
        cls = self.classpool.getModelClass(name)
        model = cls(args)

        self.models[keymap] = model

        return model







