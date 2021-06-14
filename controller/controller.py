

"""


"""

from proto import *
from trainer import *
class Controller:

    def __init__(self, proto):

        self.proto = proto
        self.trainer = Trainer()

    def call(self):
        # 解包
        self.trainer()

