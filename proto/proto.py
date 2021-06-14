

"""
    @Author: Junjie Jin
    @Date: 2021/6/9
    @Description: 协议模块
"""


class Proto:

    def __init__(self, data, model_index, status):
        """
        协议模块
        :param data:
        :param model_index:
        :param status:
        """
        self.data = data
        self.model_name = model_index
        self.status = status


class ReturnType:

    def __init__(self, data, model_index, status, loss, date, remain, **metrics):

        """
        返回类型
        :param data: 执行结果
        :param model_index: 模型索引
        :param status: 模型状态
        :param loss: 损失
        :param date: 当前时间
        :param remain: 剩余时间
        :param metrics: 指标json
        """

        self.data = data
        self.model_name = model_index
        self.status = status
        self.loss = loss
        self.date = date
        self.remain = remain
        self.metrics = metrics

    def to_description(self):
        """
        返回描述
        :return: 返回字典描述
        """
        return {
            "data": self.data,
            "model_name": self.model_name,
            "status": self.status,
            "loss": self.loss,
            "date": self.date,
            "remain": self.remain,
            "metrics": self.metrics
        }

