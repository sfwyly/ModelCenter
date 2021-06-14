

"""

    @Author: Junjie Jin
    @Date: 2021/6/12
    @Description:
    调度中心
    训练中心调度已有的模型进行训练
    适配数据到对应的模型和训练中心进行训练
"""
from models.factory import ModelPool, ClassPool
from trainer import Trainer


class Dispatcher:

    def __init__(self, modelpool, data, params):
        """
        调度中心
        :param modelpool: 模型池
        :param data: 传参数据
        :param params: 参数字典
        """
        self.modelpool = modelpool
        self.data = data
        self.params = params
        self.trainer = Trainer()

    def dispatcher(self, name=None):
        """
        任务调度
        :return:
        """
        model = self.modelpool.getModel(name, params)



        result = trainer.train(model)

        return model


if __name__ == "__main__":
    classpool = ClassPool()

    modelpool = ModelPool(classpool)

    modelname = "MLP"
    params = {
        'input_dim': 3,
        'hidden_dim': 8,
        'output_dim': 1,
        'activation': 'relu',
        'bias': True,
        'loss_list': ['MSE']  # 损失列表
    }
    params2 = {
        'input_dim': 3,
        'hidden_dim': 8,
        'output_dim': 2,
        'activation': 'relu',
        'bias': True,
        'loss_list': ['MSE']  # 损失列表
    }

    model = modelpool.getModel(modelname, params)
    model2 = modelpool.getModel(modelname, params2)

    print(model, model2)
    # train_params ={}
    # trainer = Trainer(model, train_params)
    #
    # # 单步训练给结果
    # metrics = trainer.train()

