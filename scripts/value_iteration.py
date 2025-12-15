import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

from scripts.tools import grid_env


class Value_iteration:
    def __init__(self, env: grid_env.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        # 初始策略是均匀分配 这里5个动作都是0.2
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象


    def value_iteration(self, tolerance=0.001, steps=100):
        """
        迭代求解最优贝尔曼公式 得到 最优state value tolerance 和 steps 满足其一即可
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止 建议将此变量设置大一些
        :return: 剩余迭代次数
        """
        state_value_k = np.ones(self.state_space_size)
        # state value 不变判定 到达不动点
        while np.linalg.norm(state_value_k - self.state_value, ord=1) > tolerance and steps > 0:
            steps -= 1
            self.state_value = state_value_k.copy()
            # 这里只进行一步迭代 这里给的是初始化 state value
            self.policy, state_value_k = self.policy_improvement(state_value_k.copy())
        return steps