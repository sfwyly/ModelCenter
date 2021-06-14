

"""
    Readme: 每个文件必须注明下述注释字段， 代码编写者， 时间， 作用描述
    @Author: Junjie Jin
    @Date: 2021/6/9
    @Description: 该类用于表示感知器

"""
import torch
import torch.nn as nn
from utils import getLossClass


class MLP:

    def __init__(self, kwargs):

        """
        必须添加初始化参数字段注释， 并声明类型
        :param intput_dim: 输入维度 int
        :param hidden_dim: 隐藏层维度 int
        :param output_dim: 输出维度 int
        :param activation: 激活函数 str
        :param bias: 偏差 bool
        :param loss_list: 损失列表 字符串
        """

        self.input_dim = kwargs['input_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.output_dim = kwargs['output_dim']
        self.activation = kwargs['activation']
        self.bias = kwargs['bias']

        self.loss_list = kwargs['loss_list']
        # 请过滤
        self.LossClass = getLossClass(self.loss_list)
        self.model = None

    def build(self):
        """
        该函数构建真实的MLP类
        :return: 返回真实构建好的MLP类， 这里的模型必须返回并最终会注册到工厂中心
        """

        # 这里的每个参数请判断异常后再传参
        self.model = MLPClass(input_dim= self.input_dim, hidden_dim= self.hidden_dim, output_dim= self.output_dim, activation= self.activation, bias= self.bias)

        return self.model

    @classmethod
    def to_description(cls):
        """
        返回参数列表字符串 json
        :return:
        """
        params = {
            'intput_dim': '输入维度 int',
            'hidden_dim': '隐藏层维度 int',
            'output_dim': '输出维度 int',
            'activation': '激活函数 str',
            'bias': '偏差 bool',
            'loss_list': '损失列表 list'
        }

        return params

    def cal_loss(self, x_true, x_pred):
        """
        根据输入数据计算损失
        :param x_true:
        :param x_pred:
        :return:
        """
        loss = 0.
        for lc in self.LossClass:
            lc = lc(is_torch= True) # 初始化， 参数自定义
            loss += lc(x_true, x_pred)

        return loss


class MLPClass(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str, bias: bool):

        """
        必须添加初始化参数字段注释， 并声明类型
        :param intput_dim: 输入维度 int
        :param hidden_dim: 隐藏层维度 int
        :param output_dim: 输出维度 int
        :param activation: 激活函数 str
        :param bias: 偏差 bool
        """

        super(MLPClass, self).__init__()

        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=bias)

        if activation == "relu":
            self.activ = nn.ReLU()
        elif activation == "prelu":
            self.activ = nn.PReLU()
        elif activation == "elu":
            self.activ = nn.ELU()
        elif activation == "lrelu":
            self.activ = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activ = nn.Sigmoid()
        elif activation == "tanh":
            self.activ = nn.Tanh()
        else:
            self.activ = nn.ReLU()

        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=bias)

    def forward(self, x):

        """
        :param x: 输入
        :return: 输出结果
        """
        x = self.linear1(x)
        x = self.activ(x)
        x = self.linear2(x)
        x = self.activ(x)

        # 严禁这么写在一行
        # return self.activ(self.linear2(self.activ(self.linear1)))

        return x


if __name__ == '__main__':

    mlp = MLPClass(input_dim=3, output_dim=1, hidden_dim=8, bias=True, activation="relu")

    input = torch.randn((4,3),requires_grad=True)
    output = mlp(input)

    labels = torch.ones((4,1))

    loss = torch.mean(torch.abs(labels - output))

    loss.backward()

    print(input.grad)
    print(mlp.linear1.weight.grad)
